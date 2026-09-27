"""Small Qwen3 MoE regression tests; no model downloads are required.

Run with ``pytest tests/models/decoder_only/language/test_qwen3_moe.py`` on a
CUDA vLLM installation. The full-model HF comparison additionally requires a
Transformers version with Qwen3 MoE (for example, 4.51.3).
"""

import os

import pytest
import ray
import torch
from torch import nn
from torch.nn import functional as F

from tests.utils import (init_test_distributed_environment,
                         multi_process_parallel)
from vllm.config import LoRAConfig
from vllm.distributed import differentiable_identity
from vllm.lora.layers import MergedQKVParallelLinearWithLora
from vllm.model_executor.models.qwen3_moe import (Qwen3MoeAttention,
                                                  Qwen3MoeForCausalLM,
                                                  Qwen3MoeMLP,
                                                  Qwen3MoeSparseMoeBlock,
                                                  _GatherTrainingLogits)
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.qwen3_moe import Qwen3MoeConfig


def tiny_config(**overrides):
    values = dict(
        architectures=["Qwen3MoeForCausalLM"],
        vocab_size=97,  # Deliberately not a padded vocabulary size.
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,  # Deliberately different from hidden_size // heads.
        max_position_embeddings=128,
        moe_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        attention_dropout=0.0,
        attention_bias=False,
        use_sliding_window=False,
    )
    values.update(overrides)
    if values.get("head_dim") is None:
        values.pop("head_dim", None)
    return Qwen3MoeConfig(**values)


@pytest.fixture
def single_gpu(request):
    if not torch.cuda.is_available():
        pytest.skip("Qwen3 MoE layers require a CUDA vLLM installation")
    request.getfixturevalue("dist_init")
    torch.manual_seed(7)
    with torch.device("cuda"):
        yield


@pytest.mark.parametrize("head_dim", [None, 64])
def test_config_round_trip(tmp_path, head_dim):
    """Exercise vLLM's registry on both old and new Transformers versions."""
    config = tiny_config(head_dim=head_dim)
    config.save_pretrained(tmp_path)
    restored = get_config(tmp_path, trust_remote_code=False)
    assert isinstance(restored, Qwen3MoeConfig)
    assert restored.model_type == "qwen3_moe"
    actual_head_dim = getattr(
        restored, "head_dim",
        restored.hidden_size // restored.num_attention_heads)
    expected_head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads)
    assert actual_head_dim == expected_head_dim
    if head_dim is not None:
        assert actual_head_dim == head_dim
    assert restored.num_experts == 4
    assert restored.num_experts_per_tok == 2


def test_sparse_layer_placement(single_gpu):
    config = tiny_config(num_hidden_layers=4,
                         decoder_sparse_step=2,
                         mlp_only_layers=[1])
    model = Qwen3MoeForCausalLM(config)
    assert [type(layer.mlp) for layer in model.model.layers] == [
        Qwen3MoeMLP, Qwen3MoeMLP, Qwen3MoeMLP, Qwen3MoeSparseMoeBlock
    ]
    attn = model.model.layers[0].self_attn
    assert attn.head_dim == 64
    assert attn.qkv_proj.weight.shape == (512, 128)
    assert attn.o_proj.weight.shape == (128, 256)
    assert attn.q_norm.weight.shape == (64, )
    assert attn.k_norm.weight.shape == (64, )


def test_expert_and_attention_weight_loading(single_gpu):
    config = tiny_config(num_hidden_layers=1)
    model = Qwen3MoeForCausalLM(config)
    tensors = {}
    for expert_id in range(config.num_experts):
        prefix = f"model.layers.0.mlp.experts.{expert_id}"
        for projection in ("gate_proj", "up_proj", "down_proj"):
            shape = ((config.hidden_size, config.moe_intermediate_size)
                     if projection == "down_proj" else
                     (config.moe_intermediate_size, config.hidden_size))
            tensors[f"{prefix}.{projection}.weight"] = torch.randn(shape)

    attn_prefix = "model.layers.0.self_attn"
    for name, width in (("q_proj", 256), ("k_proj", 128), ("v_proj", 128)):
        tensors[f"{attn_prefix}.{name}.weight"] = torch.randn(width, 128)
    tensors[f"{attn_prefix}.q_norm.weight"] = torch.randn(64)
    tensors[f"{attn_prefix}.k_norm.weight"] = torch.randn(64)
    tensors["model.layers.0.mlp.gate.weight"] = torch.randn(4, 128)
    model.load_weights(tensors.items())

    layer = model.model.layers[0]
    for expert_id in range(config.num_experts):
        prefix = f"model.layers.0.mlp.experts.{expert_id}"
        expected = torch.cat((tensors[f"{prefix}.gate_proj.weight"],
                              tensors[f"{prefix}.up_proj.weight"]))
        torch.testing.assert_close(layer.mlp.experts.w13_weight[expert_id],
                                   expected)
        torch.testing.assert_close(layer.mlp.experts.w2_weight[expert_id],
                                   tensors[f"{prefix}.down_proj.weight"])
    expected_qkv = torch.cat([
        tensors[f"{attn_prefix}.{name}.weight"]
        for name in ("q_proj", "k_proj", "v_proj")
    ])
    torch.testing.assert_close(layer.self_attn.qkv_proj.weight, expected_qkv)
    for name in ("q_norm", "k_norm"):
        torch.testing.assert_close(getattr(layer.self_attn, name).weight,
                                   tensors[f"{attn_prefix}.{name}.weight"])
    torch.testing.assert_close(layer.mlp.gate.weight,
                               tensors["model.layers.0.mlp.gate.weight"])


def test_inference_qk_normalization_precedes_rope(single_gpu, monkeypatch):
    config = tiny_config()
    attention = Qwen3MoeAttention(config)
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.normal_(0, 0.2)
    inputs = torch.randn(5, config.hidden_size)
    positions = torch.tensor([1, 3, 4, 7, 9])
    q, k, v = F.linear(inputs, attention.qkv_proj.weight).split(
        [attention.q_size, attention.kv_size, attention.kv_size], dim=-1)

    def reference_norm(states, weight):
        shape = states.shape
        states = states.reshape(5, -1, config.head_dim)
        states = states * torch.rsqrt(states.square().mean(-1, keepdim=True) +
                                     config.rms_norm_eps)
        return (states * weight).reshape(shape)

    expected_q, expected_k = attention.rotary_emb.forward_native(
        positions, reference_norm(q, attention.q_norm.weight),
        reference_norm(k, attention.k_norm.weight))
    captured = {}

    def capture_attention(query, key, value, kv_cache, attn_metadata):
        captured.update(q=query.clone(), k=key.clone(), v=value.clone())
        return torch.zeros_like(query)

    # Observe the paged-attention boundary without allocating a KV cache.
    monkeypatch.setattr(attention.attn, "forward", capture_attention)
    with torch.no_grad():
        attention(positions, inputs, None, None)
    torch.testing.assert_close(captured["q"], expected_q,
                               atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(captured["k"], expected_k,
                               atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(captured["v"], v, atol=2e-6, rtol=2e-5)


def dense_moe_reference(block, inputs, config):
    """Compute every expert, then select routes, independently of dispatch."""
    flat = inputs.reshape(-1, config.hidden_size)
    scores = F.linear(flat, block.gate.weight).float().softmax(-1)
    weights, indices = scores.topk(config.num_experts_per_tok, dim=-1)
    if config.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    routes = torch.zeros_like(scores).scatter(-1, indices, weights)
    gate_up = torch.einsum("th,eih->tei", flat, block.experts.w13_weight)
    gate, up = gate_up.chunk(2, dim=-1)
    outputs = torch.einsum("tei,ehi->teh", F.silu(gate) * up,
                           block.experts.w2_weight)
    return (outputs * routes.to(flat.dtype).unsqueeze(-1)).sum(1).reshape_as(
        inputs)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("grouped_gemm", [False, True])
def test_moe_outputs_and_input_gradients(single_gpu, monkeypatch, normalize,
                                         grouped_gemm):
    monkeypatch.setenv("VLLM_LMS_QWEN3_MOE_GROUPED_GEMM",
                       "1" if grouped_gemm else "0")
    config = tiny_config(norm_topk_prob=normalize)
    block = Qwen3MoeSparseMoeBlock(config)
    assert block.use_grouped_gemm == grouped_gemm
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.normal_(mean=0, std=0.05)
            parameter.requires_grad_(False)

    inputs = torch.randn(2, 5, config.hidden_size, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    expected = dense_moe_reference(block, reference_inputs, config)
    actual = block.forward_native(inputs)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    assert inputs.grad is not None and torch.count_nonzero(inputs.grad) > 0
    torch.testing.assert_close(inputs.grad, reference_inputs.grad,
                               atol=2e-5, rtol=2e-4)
    assert all(parameter.grad is None for parameter in block.parameters())

    # The serving kernel must implement the same routing convention.
    with torch.no_grad():
        fused = block(inputs.detach().reshape(-1, config.hidden_size))
    torch.testing.assert_close(fused.reshape_as(expected), expected,
                               atol=2e-4, rtol=2e-3)


class ReferenceLoRA(nn.Module):
    """Ordinary HF linear projection with the LMS Q/K/V adapter formula."""

    def __init__(self, base, lora_a, lora_b):
        super().__init__()
        self.base = base
        self.lora_a = nn.Parameter(lora_a.detach().clone())
        self.lora_b = nn.Parameter(lora_b.detach().clone())
        self.scaling = 32 / lora_a.shape[0]

    def forward(self, inputs):
        return self.base(inputs) + F.linear(
            F.linear(inputs, self.lora_a), self.lora_b) * self.scaling


@pytest.mark.parametrize("grouped_gemm", [False, True])
def test_unfused_hf_logits_lora_gradients_and_tasklets(single_gpu, monkeypatch,
                                                     grouped_gemm):
    monkeypatch.setenv("VLLM_LMS_QWEN3_MOE_GROUPED_GEMM",
                       "1" if grouped_gemm else "0")
    hf_module = pytest.importorskip(
        "transformers.models.qwen3_moe.modeling_qwen3_moe")
    # Include a dense layer as well as MoE, grouped-query attention, and an
    # attention projection dimension that differs from the residual dimension.
    config = tiny_config(mlp_only_layers=[1])
    config._attn_implementation = "eager"
    reference = hf_module.Qwen3MoeForCausalLM(config).eval()
    reference.requires_grad_(False)
    model = Qwen3MoeForCausalLM(config)
    model.load_weights(reference.state_dict().items())

    lora_config = LoRAConfig(max_lora_rank=8, lora_dtype=torch.float32)
    for layer in model.model.layers:
        qkv = MergedQKVParallelLinearWithLora(layer.self_attn.qkv_proj)
        qkv.create_lora_weights(1, lora_config, config)
        layer.self_attn.qkv_proj = qkv
    model.add_lora_train(torch.device("cuda"))

    adapters = []
    for actual_layer, reference_layer in zip(model.model.layers,
                                             reference.model.layers):
        qkv = actual_layer.self_attn.qkv_proj
        for projection in ("q_proj", "k_proj", "v_proj"):
            lora_a = getattr(qkv, f"lora_a_train_{projection}")
            lora_b = getattr(qkv, f"lora_b_train_{projection}")
            with torch.no_grad():
                lora_a.normal_(0, 0.02)
                # Nonzero B tests A gradients as well as B gradients.
                lora_b.normal_(0, 0.02)
            reference_lora = ReferenceLoRA(
                getattr(reference_layer.self_attn, projection), lora_a, lora_b)
            setattr(reference_layer.self_attn, projection, reference_lora)
            adapters.append((lora_a, lora_b, reference_lora))

    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    positions = torch.tensor([[3, 4, 5, 6], [1, 2, 3, 4]])
    inputs = torch.randn(2, 4, config.hidden_size, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    tasklets = list(model.unfused_forward(input_ids,
                                          attention_mask,
                                          positions=positions,
                                          inputs_embeds=inputs))
    assert len(tasklets) == config.num_hidden_layers + 1
    assert all(tasklet is None for tasklet in tasklets[:-1])
    actual = tasklets[-1]
    assert actual.shape == (2, 4, config.vocab_size)
    expected = reference(inputs_embeds=reference_inputs,
                         attention_mask=attention_mask,
                         position_ids=positions,
                         use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)
    probe = torch.randn_like(expected)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad,
                               atol=3e-5, rtol=3e-4)
    for lora_a, lora_b, reference_lora in adapters:
        for actual_param, reference_param in (
            (lora_a, reference_lora.lora_a),
            (lora_b, reference_lora.lora_b),
        ):
            assert actual_param.grad is not None
            assert torch.count_nonzero(actual_param.grad) > 0
            torch.testing.assert_close(actual_param.grad, reference_param.grad,
                                       atol=3e-5, rtol=3e-4)
    assert all(param.grad is None for name, param in model.named_parameters()
               if "lora_a_train" not in name and "lora_b_train" not in name)


@ray.remote(num_gpus=1, max_calls=1)
def tp_qkv_worker(tp_size, pp_size, rank, distributed_init_port):
    # Match the distributed test harness: workers need to see both GPUs.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    torch.cuda.set_device(rank)
    init_test_distributed_environment(tp_size, pp_size, rank,
                                      distributed_init_port, local_rank=rank)
    with torch.device(f"cuda:{rank}"):
        for num_kv_heads in (2, 1):
            check_tp_qkv_reference(rank, num_kv_heads)
        check_tp_vocabulary_reference(rank)
        for grouped_gemm in (False, True):
            check_tp_moe_reference(rank, grouped_gemm)


def check_tp_qkv_reference(rank, num_kv_heads):
    config = tiny_config(num_hidden_layers=1,
                         num_key_value_heads=num_kv_heads)
    model = Qwen3MoeForCausalLM(config)
    model.requires_grad_(False)
    base = model.model.layers[0].self_attn.qkv_proj
    qkv = MergedQKVParallelLinearWithLora(base)
    lora_config = LoRAConfig(max_lora_rank=8, lora_dtype=torch.float32)
    qkv.create_lora_weights(1, lora_config, config)
    model.model.layers[0].self_attn.qkv_proj = qkv
    # A CPU generator gives both workers exactly the same full weights.
    generator = torch.Generator(device="cpu").manual_seed(31)

    def random_tensor(*shape):
        return torch.randn(*shape, generator=generator,
                           device="cpu").to(base.weight.device) * 0.05

    reference_inputs = random_tensor(2, 3, config.hidden_size).requires_grad_()
    inputs = reference_inputs.detach().clone().requires_grad_()
    references = []
    for projection, heads in (("q", 4), ("k", num_kv_heads),
                               ("v", num_kv_heads)):
        width = heads * config.head_dim
        weight = random_tensor(width, config.hidden_size)
        base.weight.weight_loader(base.weight, weight, projection)
        lora_a = random_tensor(8, config.hidden_size).requires_grad_()
        lora_b = random_tensor(width, 8).requires_grad_()
        # Q is always partitioned; a single KV head is replicated on TP=2.
        replicated = projection != "q" and num_kv_heads == 1
        shard = slice(None) if replicated else slice(rank * width // 2,
                                                     (rank + 1) * width // 2)
        actual_a = nn.Parameter(lora_a.detach().clone())
        actual_b = nn.Parameter(lora_b[shard].detach().clone())
        setattr(qkv, f"lora_a_train_{projection}_proj", actual_a)
        setattr(qkv, f"lora_b_train_{projection}_proj", actual_b)
        expected = F.linear(reference_inputs, weight) + F.linear(
            F.linear(reference_inputs, lora_a), lora_b) * 4
        probe = random_tensor(*expected.shape)
        references.append((expected, probe, shard, replicated, lora_a,
                           lora_b, actual_a, actual_b))

    actual_outputs = model._unfused_qkv_projection(
        differentiable_identity(inputs), qkv)
    actual_loss = 0
    reference_loss = 0
    for actual, reference in zip(actual_outputs, references):
        expected, probe, shard, replicated, *_ = reference
        torch.testing.assert_close(actual, expected[..., shard],
                                   atol=2e-6, rtol=2e-5)
        # Replicas receive different downstream gradients, whose sum is the
        # unsharded reference gradient; averaging them would be incorrect.
        replica_weight = (rank + 1) / 3 if replicated else 1
        actual_loss = actual_loss + (actual * probe[..., shard]).sum() * (
            replica_weight)
        reference_loss = reference_loss + (expected * probe).sum()
    actual_loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad,
                               atol=2e-6, rtol=2e-5)
    for _, _, shard, _, lora_a, lora_b, actual_a, actual_b in references:
        torch.testing.assert_close(actual_a.grad, lora_a.grad,
                                   atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(actual_b.grad, lora_b.grad[shard],
                                   atol=2e-6, rtol=2e-5)


def check_tp_vocabulary_reference(rank):
    # Both ranks compute the same full-vocabulary loss. Gathering logits must
    # not introduce a factor of TP into any gradient, including input grads.
    generator = torch.Generator(device="cpu").manual_seed(47)
    device = torch.device(f"cuda:{rank}")
    reference_inputs = torch.randn(2, 3, 16, generator=generator,
                                    device="cpu").to(device).requires_grad_()
    weight = torch.randn(128, 16, generator=generator,
                         device="cpu").to(device)
    inputs = reference_inputs.detach().clone().requires_grad_()
    shard = slice(rank * 64, (rank + 1) * 64)
    local_logits = F.linear(differentiable_identity(inputs), weight[shard])
    local_logits.retain_grad()
    actual = _GatherTrainingLogits.apply(local_logits)[..., :97]
    reference_logits = F.linear(reference_inputs, weight)
    reference_logits.retain_grad()
    expected = reference_logits[..., :97]
    assert actual.shape == (2, 3, 97)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(local_logits.grad,
                               reference_logits.grad[..., shard])
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)


def check_tp_moe_reference(rank, grouped_gemm):
    """Expert intermediate shards must sum outputs and upstream gradients."""
    os.environ["VLLM_LMS_QWEN3_MOE_GROUPED_GEMM"] = (
        "1" if grouped_gemm else "0")
    config = tiny_config(num_hidden_layers=1)
    block = Qwen3MoeSparseMoeBlock(config)
    block.requires_grad_(False)
    generator = torch.Generator(device="cpu").manual_seed(53)
    device = torch.device(f"cuda:{rank}")

    def random_tensor(*shape):
        return torch.randn(*shape, generator=generator,
                           device="cpu").to(device) * 0.05

    intermediate = config.moe_intermediate_size
    w13 = random_tensor(config.num_experts, 2 * intermediate,
                        config.hidden_size)
    w2 = random_tensor(config.num_experts, config.hidden_size, intermediate)
    router = random_tensor(config.num_experts, config.hidden_size)
    shard = slice(rank * intermediate // 2, (rank + 1) * intermediate // 2)
    with torch.no_grad():
        gate, up = w13.chunk(2, dim=1)
        block.experts.w13_weight.copy_(torch.cat((gate[:, shard],
                                                  up[:, shard]), dim=1))
        block.experts.w2_weight.copy_(w2[:, :, shard])
        block.gate.weight.copy_(router)

    reference_inputs = random_tensor(7, config.hidden_size).requires_grad_()
    inputs = reference_inputs.detach().clone().requires_grad_()
    scores = F.linear(reference_inputs, router).softmax(-1)
    weights, indices = scores.topk(config.num_experts_per_tok, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    routes = torch.zeros_like(scores).scatter(-1, indices, weights)
    gate, up = torch.einsum("th,eih->tei", reference_inputs, w13).chunk(
        2, dim=-1)
    expert_outputs = torch.einsum("tei,ehi->teh", F.silu(gate) * up, w2)
    expected = (expert_outputs * routes.unsqueeze(-1)).sum(1)
    actual = block.forward_native(differentiable_identity(inputs))
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    probe = random_tensor(*expected.shape)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad,
                               atol=2e-6, rtol=2e-4)
    assert all(parameter.grad is None for parameter in block.parameters())


@pytest.mark.distributed_2_gpus
@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Qwen3 MoE TP gradients require two CUDA GPUs")
def test_tp_qkv_moe_and_vocabulary_gradients():
    multi_process_parallel(2, 1, tp_qkv_worker)
