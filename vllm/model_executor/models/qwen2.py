# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
import math
from typing import Generator, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen2Config

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (differentiable_all_gather,
                              differentiable_all_reduce_sum,
                              differentiable_identity, get_pp_group,
                              get_tensor_model_parallel_world_size)
from vllm.lora.layers import MergedQKVParallelLinearWithLora
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers)


class Qwen2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen2Attention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 rope_scaling: Optional[Tuple] = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
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
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling)
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2Model(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen2DecoderLayer(config=config,
                                             cache_config=cache_config,
                                             quant_config=quant_config),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        # LLMStation only uses tensor parallelism and deep-copies the model
        # before starting its fine-tuning workers. The local factory function
        # cannot be serialized across those process boundaries.
        self.make_empty_intermediate_tensors = None
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)


class Qwen2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        # TODO (@robertgshaw2): see if this can be moved out
        if (cache_config.sliding_window is not None
                and hasattr(config, "max_window_layers")):
            raise ValueError("Sliding window for some but all layers is not "
                             "supported. This model uses sliding window "
                             "but `max_window_layers` = %s is less than "
                             "`num_hidden_layers` = %s. Please open an issue "
                             "to discuss this feature." % (
                                 config.max_window_layers,
                                 config.num_hidden_layers,
                             ))

        super().__init__()

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen2Model(config, cache_config, quant_config)

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size,
                                          quant_config=quant_config)

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
        self.make_empty_intermediate_tensors = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        loader.load_weights(weights)

    def add_lora_train(self, device: torch.device) -> None:
        """Add the trainable Q/K/V LoRA weights used by LLMStation."""
        for param in self.parameters():
            param.requires_grad_(False)

        for module in self.modules():
            if not isinstance(module, MergedQKVParallelLinearWithLora):
                continue

            # Match the initialization and scaling used by the Llama LMS
            # path. Resetting the seed also keeps all TP workers in sync.
            torch.manual_seed(0)
            rank = module.lora_config.max_lora_rank
            module.lora_a_train_q_proj = nn.Parameter(
                torch.empty(rank, module.input_size, device=device))
            nn.init.kaiming_uniform_(module.lora_a_train_q_proj,
                                     a=math.sqrt(5))
            module.lora_b_train_q_proj = nn.Parameter(
                torch.zeros(module.q_proj_shard_size, rank, device=device))

            module.lora_a_train_k_proj = nn.Parameter(
                torch.empty(rank, module.input_size, device=device))
            nn.init.kaiming_uniform_(module.lora_a_train_k_proj,
                                     a=math.sqrt(5))
            module.lora_b_train_k_proj = nn.Parameter(
                torch.zeros(module.kv_proj_shard_size, rank, device=device))

            module.lora_a_train_v_proj = nn.Parameter(
                torch.empty(rank, module.input_size, device=device))
            nn.init.kaiming_uniform_(module.lora_a_train_v_proj,
                                     a=math.sqrt(5))
            module.lora_b_train_v_proj = nn.Parameter(
                torch.zeros(module.kv_proj_shard_size, rank, device=device))

    def _unfused_qkv_projection(
        self,
        hidden_states: torch.Tensor,
        lora_layer: MergedQKVParallelLinearWithLora,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run Qwen2's biased Q/K/V projections with trainable LoRA."""
        q_size = lora_layer.q_proj_shard_size
        kv_size = lora_layer.kv_proj_shard_size
        q_weight, k_weight, v_weight = lora_layer.base_layer.weight.split(
            (q_size, kv_size, kv_size), dim=0)

        bias = lora_layer.base_layer.bias
        if bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = bias.split((q_size, kv_size, kv_size),
                                                dim=0)

        query_states = F.linear(hidden_states, q_weight, q_bias)
        key_states = F.linear(hidden_states, k_weight, k_bias)
        value_states = F.linear(hidden_states, v_weight, v_bias)

        scaling = 32 / lora_layer.lora_config.max_lora_rank
        projections = (
            (query_states, lora_layer.lora_a_train_q_proj,
             lora_layer.lora_b_train_q_proj),
            (key_states, lora_layer.lora_a_train_k_proj,
             lora_layer.lora_b_train_k_proj),
            (value_states, lora_layer.lora_a_train_v_proj,
             lora_layer.lora_b_train_v_proj),
        )
        outputs = []
        for base_output, lora_a, lora_b in projections:
            after_a = F.linear(hidden_states.to(lora_a.dtype), lora_a)
            lora_output = F.linear(after_a, lora_b) * scaling
            outputs.append((base_output + lora_output).to(base_output.dtype))

        return outputs[0], outputs[1], outputs[2]

    @staticmethod
    def _repeat_kv(hidden_states: torch.Tensor,
                   num_repeats: int) -> torch.Tensor:
        if num_repeats == 1:
            return hidden_states
        batch_size, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch_size, num_kv_heads, num_repeats, seq_len, head_dim)
        return hidden_states.reshape(batch_size,
                                     num_kv_heads * num_repeats, seq_len,
                                     head_dim)

    def unfused_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Generator[Union[torch.Tensor, IntermediateTensors], None, None]:
        """Run a differentiable Qwen2 forward pass one layer at a time."""
        if not get_pp_group().is_first_rank:
            raise NotImplementedError(
                "Qwen2 LMS fine-tuning does not support pipeline parallelism")

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            # Unlike Llama, Qwen2 does not expose embedding LoRA modules, so
            # this layer is not wrapped and has no ``base_layer`` attribute.
            hidden_states = self.model.embed_tokens(input_ids)

        batch_size, seq_len = input_ids.shape
        if positions is None:
            positions = torch.arange(seq_len,
                                     dtype=torch.long,
                                     device=input_ids.device)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
        elif positions.dim() == 1:
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

            residual = hidden_states
            hidden_states = layer.input_layernorm.forward_native(
                hidden_states)
            hidden_states = differentiable_identity(hidden_states)

            query_states, key_states, value_states = (
                self._unfused_qkv_projection(hidden_states,
                                             layer.self_attn.qkv_proj))
            rotary_emb = getattr(layer.self_attn.rotary_emb, "base_layer",
                                 layer.self_attn.rotary_emb)
            query_states, key_states = rotary_emb.forward_native(
                positions, query_states, key_states)

            query_states = query_states.view(
                batch_size, seq_len, layer.self_attn.num_heads,
                layer.self_attn.head_dim).transpose(1, 2)
            key_states = key_states.view(
                batch_size, seq_len, layer.self_attn.num_kv_heads,
                layer.self_attn.head_dim).transpose(1, 2)
            value_states = value_states.view(
                batch_size, seq_len, layer.self_attn.num_kv_heads,
                layer.self_attn.head_dim).transpose(1, 2)

            num_kv_groups = (layer.self_attn.num_heads //
                             layer.self_attn.num_kv_heads)
            key_states = self._repeat_kv(key_states, num_kv_groups)
            value_states = self._repeat_kv(value_states, num_kv_groups)

            attn_weights = (
                torch.matmul(query_states, key_states.transpose(2, 3)) *
                layer.self_attn.scaling)
            if causal_mask is not None:
                attn_weights = (
                    attn_weights +
                    causal_mask[:, :, :, :key_states.shape[-2]])
            attn_weights = F.softmax(attn_weights,
                                     dim=-1,
                                     dtype=torch.float32).to(
                                         query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            expected_shape = (batch_size, layer.self_attn.num_heads, seq_len,
                              layer.self_attn.head_dim)
            if attn_output.shape != expected_shape:
                raise ValueError("Attention output has shape "
                                 f"{tuple(attn_output.shape)}; expected "
                                 f"{expected_shape}")
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(
                batch_size, seq_len, -1)

            hidden_states = (
                layer.self_attn.o_proj.base_layer.quant_method.apply(
                    layer.self_attn.o_proj.base_layer, attn_output))
            hidden_states = differentiable_all_reduce_sum(hidden_states)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm.forward_native(
                hidden_states)
            hidden_states = differentiable_identity(hidden_states)

            partition_size = (
                layer.mlp.gate_up_proj.base_layer.output_partition_sizes[0])
            gate_states = F.linear(
                hidden_states,
                layer.mlp.gate_up_proj.base_layer.weight[:partition_size])
            up_states = F.linear(
                hidden_states,
                layer.mlp.gate_up_proj.base_layer.weight[partition_size:])
            hidden_states = F.silu(gate_states) * up_states
            hidden_states = layer.mlp.down_proj.base_layer.quant_method.apply(
                layer.mlp.down_proj.base_layer, hidden_states)
            hidden_states = differentiable_all_reduce_sum(hidden_states)
            hidden_states = residual + hidden_states

            # Each layer is one forward tasklet and therefore a preemption
            # point for inference work.
            yield

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual,
            })

        hidden_states = self.model.norm.forward_native(hidden_states)
        logits = F.linear(hidden_states, self.lm_head.weight).float()
        if self.lm_head.tp_size > 1:
            logits = differentiable_all_gather(logits)
        logits = logits[:, :, :self.model.config.vocab_size]
        yield logits
