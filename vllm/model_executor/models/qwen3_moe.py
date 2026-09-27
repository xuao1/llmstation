# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3 MoE inference and layer-wise Q/K/V LoRA training for LLMStation.

The inference path uses packed, tensor-parallel experts. The training path
uses the same frozen weights with native PyTorch operations so gradients can
flow through both the selected experts and their routing probabilities.
"""
from typing import Generator, Iterable, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (differentiable_all_reduce_sum,
                              differentiable_identity, get_pp_group,
                              get_tensor_model_parallel_world_size,
                              get_tp_group)
from vllm.lora.layers import MergedQKVParallelLinearWithLora
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.sequence import IntermediateTensors

from .qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM, Qwen2MLP, Qwen2Model
from .utils import PPMissingLayer, is_pp_missing_parameter, make_layers


class _GatherTrainingLogits(torch.autograd.Function):
    """Gather vocabulary shards for the identical loss computed on each rank."""

    @staticmethod
    def forward(ctx, logits):
        group = get_tp_group()
        ctx.rank = group.rank_in_group
        ctx.shard_size = logits.shape[-1]
        return group.all_gather(logits, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        # Every rank computes the same loss; summing copies here would scale
        # every model gradient by the TP world size.
        return grad_output.narrow(-1, ctx.rank * ctx.shard_size,
                                  ctx.shard_size).contiguous()


class _ReplicatedKVLoRAWeight(torch.autograd.Function):
    """Synchronize K/V B gradients only among replicas of the same KV head."""

    @staticmethod
    def forward(ctx, weight, shard_id, num_shards):
        ctx.shard_id = shard_id
        ctx.num_shards = num_shards
        return weight

    @staticmethod
    def backward(ctx, grad_output):
        gradients = grad_output.new_zeros((ctx.num_shards, ) +
                                          grad_output.shape)
        gradients[ctx.shard_id].copy_(grad_output)
        torch.distributed.all_reduce(gradients,
                                     group=get_tp_group().device_group)
        return gradients[ctx.shard_id], None, None


class Qwen3MoeMLP(Qwen2MLP):

    def forward_native(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = F.linear(hidden_states,
                            self.gate_up_proj.weight).chunk(2, dim=-1)
        hidden_states = F.silu(gate) * up
        hidden_states = F.linear(hidden_states, self.down_proj.weight)
        return differentiable_all_reduce_sum(hidden_states)


class Qwen3MoeSparseMoeBlock(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.quant_config = quant_config
        tp_size = get_tensor_model_parallel_world_size()
        if config.hidden_act != "silu":
            raise ValueError("Qwen3 MoE only supports the silu activation")
        if not 0 < self.top_k <= self.num_experts:
            raise ValueError("num_experts_per_tok must be in [1, num_experts]")
        if config.moe_intermediate_size % tp_size:
            raise ValueError("moe_intermediate_size must be divisible by the "
                             "tensor parallel size")

        self.gate = ReplicatedLinear(config.hidden_size,
                                     self.num_experts,
                                     bias=False,
                                     quant_config=None)
        self.experts = FusedMoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=True,
            renormalize=self.norm_topk_prob,
            quant_config=quant_config,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, original_shape[-1])
        router_logits, _ = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states=hidden_states,
                                     router_logits=router_logits)
        return hidden_states.view(original_shape)

    def forward_native(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Differentiable routing and expert evaluation on local TP shards."""
        if self.quant_config is not None:
            raise NotImplementedError(
                "Qwen3 MoE LMS fine-tuning requires unquantized weights")
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, original_shape[-1])
        router_logits = F.linear(hidden_states, self.gate.weight)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights,
                                                       self.top_k,
                                                       dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        output = torch.zeros_like(hidden_states)
        for expert_idx in range(self.num_experts):
            token_idx, slot_idx = torch.where(selected_experts == expert_idx)
            if token_idx.numel() == 0:
                continue
            expert_input = hidden_states[token_idx]
            gate, up = F.linear(
                expert_input,
                self.experts.w13_weight[expert_idx]).chunk(2, dim=-1)
            expert_output = F.linear(F.silu(gate) * up,
                                      self.experts.w2_weight[expert_idx])
            expert_output = expert_output * routing_weights[token_idx,
                                                             slot_idx, None]
            output.index_add_(0, token_idx, expert_output)

        # Each rank holds an intermediate-dimension shard of every expert.
        # Input gradients are reduced by the caller's differentiable_identity.
        output = differentiable_all_reduce_sum(output)
        return output.view(original_shape)


class Qwen3MoeAttention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp_size == 0
        assert self.total_num_heads % self.total_num_kv_heads == 0
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(config, "head_dim",
                                config.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        attention_bias = getattr(config, "attention_bias", False)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(q.shape)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3MoeDecoderLayer(Qwen2DecoderLayer):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        nn.Module.__init__(self)
        self.self_attn = Qwen3MoeAttention(config, cache_config, quant_config)
        mlp_only_layers = getattr(config, "mlp_only_layers", []) or []
        if (layer_idx not in mlp_only_layers and config.num_experts > 0
                and (layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3MoeSparseMoeBlock(config, quant_config)
        else:
            self.mlp = Qwen3MoeMLP(config.hidden_size, config.intermediate_size,
                                   config.hidden_act, quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)


class Qwen3MoeModel(Qwen2Model):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size,
                quant_config=quant_config)
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen3MoeDecoderLayer(
                config, int(prefix.split(".")[-1]), cache_config, quant_config),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        # A bound method, unlike a local factory closure, survives the LMS
        # model deepcopy / multiprocessing transfer.
        return IntermediateTensors({
            key: torch.zeros((batch_size, self.config.hidden_size),
                             dtype=dtype, device=device)
            for key in ("hidden_states", "residual")
        })

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if is_pp_missing_parameter(name, self):
                continue
            # Experts must be mapped before ordinary gate/up projection names.
            for param_name, weight_name, expert_id, shard_id in (
                    expert_params_mapping):
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if (name.endswith((".bias", "_bias"))
                        and name not in params_dict):
                    break
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, name,
                                    shard_id=shard_id, expert_id=expert_id)
                break
            else:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name.endswith(".bias") and name not in params_dict:
                        break
                    param = params_dict[name]
                    param.weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)


class Qwen3MoeForCausalLM(Qwen2ForCausalLM):
    # FusedMoE has no LoRA wrapper in this vLLM version. Keep adapter loading
    # on attention projections, with LMS training Q/K/V just like Qwen2.
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    supported_lora_modules = ["qkv_proj", "o_proj"]
    embedding_modules = {}
    embedding_padding_modules = []
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        nn.Module.__init__(self)
        if ((cache_config is not None
             and cache_config.sliding_window is not None)
                or (getattr(config, "use_sliding_window", False)
                    and getattr(config, "sliding_window", None) is not None)):
            raise ValueError("Qwen3 MoE sliding-window attention is not "
                             "supported by this implementation")
        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        self.model = Qwen3MoeModel(config, cache_config, quant_config)
        if not get_pp_group().is_last_rank:
            self.lm_head = PPMissingLayer()
        elif config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size,
                                          quant_config=quant_config)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def add_lora_train(self, device: torch.device) -> None:
        if self.quant_config is not None:
            raise NotImplementedError(
                "Qwen3 MoE LMS fine-tuning requires unquantized weights")
        if not (get_pp_group().is_first_rank and get_pp_group().is_last_rank):
            raise NotImplementedError(
                "Qwen3 MoE LMS fine-tuning does not support pipeline "
                "parallelism")
        for layer in self.model.layers:
            if not isinstance(layer.self_attn.qkv_proj,
                              MergedQKVParallelLinearWithLora):
                raise ValueError("Enable LoRA before starting Qwen3 MoE LMS "
                                 "fine-tuning")
        super().add_lora_train(device)
        self.train()

    def _unfused_qkv_projection(
        self,
        hidden_states: torch.Tensor,
        lora_layer: MergedQKVParallelLinearWithLora,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sizes = (lora_layer.q_proj_shard_size,
                 lora_layer.kv_proj_shard_size,
                 lora_layer.kv_proj_shard_size)
        base = lora_layer.base_layer
        weights = base.weight.split(sizes, dim=0)
        biases = ((None, None, None) if base.bias is None else
                  base.bias.split(sizes, dim=0))
        scaling = 32 / lora_layer.lora_config.max_lora_rank
        outputs = []
        for projection, weight, bias in zip(("q_proj", "k_proj", "v_proj"),
                                             weights, biases):
            output = F.linear(hidden_states, weight, bias)
            lora_a = getattr(lora_layer, f"lora_a_train_{projection}")
            lora_b = getattr(lora_layer, f"lora_b_train_{projection}")
            # A is replicated even when the projection's output is sharded.
            lora_a = differentiable_identity(lora_a)
            if projection != "q_proj" and base.num_kv_head_replicas > 1:
                lora_b = _ReplicatedKVLoRAWeight.apply(
                    lora_b, lora_layer.kv_shard_id,
                    base.total_num_kv_heads)
            after_a = F.linear(hidden_states.to(lora_a.dtype), lora_a)
            lora_output = F.linear(after_a, lora_b) * scaling
            outputs.append((output + lora_output).to(output.dtype))
        return outputs[0], outputs[1], outputs[2]

    def unfused_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        """Yield after each decoder layer, then yield the training logits."""
        if self.quant_config is not None:
            raise NotImplementedError(
                "Qwen3 MoE LMS fine-tuning requires unquantized weights")
        if not (get_pp_group().is_first_rank and get_pp_group().is_last_rank):
            raise NotImplementedError(
                "Qwen3 MoE LMS fine-tuning does not support pipeline "
                "parallelism")
        hidden_states = (inputs_embeds if inputs_embeds is not None else
                         self.model.embed_tokens(input_ids))
        batch_size, seq_len = input_ids.shape
        if positions is None:
            positions = torch.arange(seq_len, dtype=torch.long,
                                     device=input_ids.device)
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(batch_size, -1)

        from transformers.modeling_attn_mask_utils import (
            _prepare_4d_causal_attention_mask)
        causal_mask = _prepare_4d_causal_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_len),
            inputs_embeds=hidden_states,
            past_key_values_length=0,
        )

        for layer_idx in range(self.model.start_layer, self.model.end_layer):
            layer = self.model.layers[layer_idx]
            attn = layer.self_attn
            residual = hidden_states
            hidden_states = layer.input_layernorm.forward_native(hidden_states)
            hidden_states = differentiable_identity(hidden_states)
            q, k, v = self._unfused_qkv_projection(hidden_states, attn.qkv_proj)
            q = attn.q_norm.forward_native(
                q.reshape(batch_size, seq_len, attn.num_heads,
                          attn.head_dim)).reshape(batch_size, seq_len, -1)
            k = attn.k_norm.forward_native(
                k.reshape(batch_size, seq_len, attn.num_kv_heads,
                          attn.head_dim)).reshape(batch_size, seq_len, -1)
            rotary_emb = getattr(attn.rotary_emb, "base_layer", attn.rotary_emb)
            q, k = rotary_emb.forward_native(positions, q, k)
            q = q.view(batch_size, seq_len, attn.num_heads,
                       attn.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, attn.num_kv_heads,
                       attn.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, attn.num_kv_heads,
                       attn.head_dim).transpose(1, 2)
            num_kv_groups = attn.num_heads // attn.num_kv_heads
            k = self._repeat_kv(k, num_kv_groups)
            v = self._repeat_kv(v, num_kv_groups)
            attn_weights = torch.matmul(q, k.transpose(2, 3)) * attn.scaling
            if causal_mask is not None:
                attn_weights = attn_weights + causal_mask
            attn_weights = F.softmax(attn_weights, dim=-1,
                                     dtype=torch.float32).to(q.dtype)
            attn_weights = F.dropout(attn_weights, p=attn.attention_dropout,
                                     training=self.training)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(
                batch_size, seq_len, -1)
            o_proj = getattr(attn.o_proj, "base_layer", attn.o_proj)
            hidden_states = F.linear(attn_output, o_proj.weight)
            hidden_states = differentiable_all_reduce_sum(hidden_states)
            if o_proj.bias is not None:
                hidden_states = hidden_states + o_proj.bias
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm.forward_native(
                hidden_states)
            hidden_states = differentiable_identity(hidden_states)
            hidden_states = layer.mlp.forward_native(hidden_states)
            hidden_states = residual + hidden_states
            yield

        hidden_states = self.model.norm.forward_native(hidden_states)
        # The vocabulary is column-sharded too: reduce hidden-state gradients
        # before backpropagating into the replicated final normalization.
        hidden_states = differentiable_identity(hidden_states)
        logits = F.linear(hidden_states, self.lm_head.weight).float()
        if self.lm_head.tp_size > 1:
            logits = _GatherTrainingLogits.apply(logits)
        yield logits[:, :, :self.config.vocab_size]
