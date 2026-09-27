"""Numerical and autograd checks for the frozen-expert training kernel.

Run ``pytest tests/kernels/test_grouped_moe.py`` on a CUDA installation.
"""

import pytest
import torch
from torch.nn import functional as F

from vllm.model_executor.layers.fused_moe.grouped_gemm import grouped_moe

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="grouped MoE requires CUDA")


@pytest.fixture(autouse=True)
def ieee_matmul():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def torch_moe_reference(inputs, w13, w2, indices, weights):
    """Explicit expert dispatch is independent of the grouped GPU schedule."""
    output = inputs * 0 + weights.sum() * 0
    for expert in range(w13.shape[0]):
        tokens, slots = torch.where(indices == expert)
        if tokens.numel() == 0:
            continue
        gate, up = F.linear(inputs[tokens], w13[expert]).chunk(2, dim=-1)
        values = F.linear(F.silu(gate) * up, w2[expert])
        output = output.index_add(0, tokens,
                                  values * weights[tokens, slots, None])
    return output


@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("top_k,normalize,skewed", [
    (1, False, True),
    (2, True, True),
    (2, False, False),
    (5, True, False),
])
def test_grouped_moe_outputs_and_gradients(dtype, top_k, normalize, skewed):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 requires a supported CUDA GPU")
    torch.manual_seed(41)
    # All dimensions have tails. Transposed input and output gradients also
    # exercise strides that differ from the common contiguous training case.
    tokens, hidden_size, intermediate_size, experts = 35, 37, 23, 5
    inputs = torch.randn(hidden_size, tokens, device="cuda", dtype=dtype)
    inputs = inputs.t().detach().requires_grad_()
    reference_inputs = inputs.detach().clone().requires_grad_()
    w13 = torch.randn(experts, 2 * intermediate_size, hidden_size,
                      device="cuda", dtype=dtype) * 0.08
    w2 = torch.randn(experts, hidden_size, intermediate_size,
                     device="cuda", dtype=dtype) * 0.08
    logits = torch.randn(tokens, experts, device="cuda")
    if skewed:
        # Only the first top_k experts are selected; all others are empty.
        logits[:, :top_k] += 10
    logits.requires_grad_()
    reference_logits = logits.detach().clone().requires_grad_()

    def routing(scores):
        weights, indices = scores.softmax(-1).topk(top_k, dim=-1)
        if normalize:
            weights = weights / weights.sum(-1, keepdim=True)
        weights = weights.to(dtype)
        weights.retain_grad()
        return weights, indices

    weights, indices = routing(logits)
    reference_weights, reference_indices = routing(reference_logits)
    torch.testing.assert_close(indices, reference_indices)
    if skewed:
        assert indices.max().item() < top_k
    actual = grouped_moe(inputs, w13, w2, indices, weights)
    expected = torch_moe_reference(reference_inputs, w13, w2,
                                   reference_indices, reference_weights)
    tolerances = {
        torch.float32: dict(atol=2e-5, rtol=2e-4),
        torch.float16: dict(atol=3e-4, rtol=1e-2),
        torch.bfloat16: dict(atol=3e-3, rtol=5e-2),
    }[dtype]
    torch.testing.assert_close(actual, expected, **tolerances)
    probe = torch.randn(hidden_size, tokens, device="cuda", dtype=dtype).t()
    assert not probe.is_contiguous()
    actual.backward(probe)
    expected.backward(probe)
    for actual_grad, expected_grad in (
        (inputs.grad, reference_inputs.grad),
        (weights.grad, reference_weights.grad),
        (logits.grad, reference_logits.grad),
    ):
        assert actual_grad is not None
        torch.testing.assert_close(actual_grad, expected_grad, **tolerances)
    assert w13.grad is None and w2.grad is None


def test_grouped_moe_empty_tokens_preserve_autograd():
    inputs = torch.empty(0, 37, device="cuda", requires_grad=True)
    weights = torch.empty(0, 2, device="cuda", requires_grad=True)
    indices = torch.empty(0, 2, device="cuda", dtype=torch.int32)
    w13 = torch.randn(5, 46, 37, device="cuda")
    w2 = torch.randn(5, 37, 23, device="cuda")
    output = grouped_moe(inputs, w13, w2, indices, weights)
    assert output.shape == inputs.shape
    output.sum().backward()
    assert inputs.grad is not None and inputs.grad.shape == inputs.shape
    assert weights.grad is not None and weights.grad.shape == weights.shape
    assert w13.grad is None and w2.grad is None


@pytest.mark.skipif(not hasattr(torch.Tensor, "coro_backward"),
                    reason="requires the LMS coroutine-enabled PyTorch build")
def test_grouped_moe_coroutine_backward():
    """Exercise the same backward entry point used by the LMS scheduler."""
    torch.manual_seed(59)
    inputs = torch.randn(19, 37, device="cuda", requires_grad=True)
    weights = torch.rand(19, 2, device="cuda", requires_grad=True)
    indices = torch.tensor([[0, 2], [2, 3]], device="cuda").repeat(10, 1)[:19]
    w13 = torch.randn(5, 46, 37, device="cuda") * 0.05
    w2 = torch.randn(5, 37, 23, device="cuda") * 0.05
    reference_inputs = inputs.detach().clone().requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    output = grouped_moe(inputs, w13, w2, indices, weights)
    expected = torch_moe_reference(reference_inputs, w13, w2, indices,
                                   reference_weights)
    torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-4)
    # Exhaust the coroutine so every backward tasklet finishes.
    for _ in output.sum().coro_backward():
        pass
    expected.sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad,
                               atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(weights.grad, reference_weights.grad,
                               atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("trainable_projection", ["w13", "w2"])
def test_grouped_moe_rejects_trainable_experts(trainable_projection):
    inputs = torch.randn(3, 37, device="cuda", requires_grad=True)
    weights = torch.ones(3, 1, device="cuda", requires_grad=True)
    indices = torch.zeros(3, 1, device="cuda", dtype=torch.int32)
    w13 = torch.randn(5, 46, 37, device="cuda")
    w2 = torch.randn(5, 37, 23, device="cuda")
    (w13 if trainable_projection == "w13" else w2).requires_grad_()
    with pytest.raises(ValueError, match="frozen"):
        grouped_moe(inputs, w13, w2, indices, weights)
