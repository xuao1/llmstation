"""Differentiable grouped expert GEMMs for attention-only MoE LoRA training.

Expert weights stay frozen. Both linear layers propagate input gradients with
the same grouped kernel and transposed weights; PyTorch handles the activation
and routing-weight gradients. This module is separate from inference MoE kernels
and is imported only when the training grouped-GEMM option is enabled.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

_BLOCK_M = 16
_BLOCK_N = 64
_BLOCK_K = 32


@triton.jit
def _grouped_gemm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    sorted_route_ids_ptr,
    expert_ids_ptr,
    num_routes_padded_ptr,
    NUM_ROUTES: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_we: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    INPUT_TOP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each M tile belongs to one expert. All experts share this launch, with
    # padding represented by sentinel route IDs rather than padded activations.
    block_m = tl.program_id(0) // tl.cdiv(N, BLOCK_N)
    block_n = tl.program_id(0) % tl.cdiv(N, BLOCK_N)
    if block_m * BLOCK_M >= tl.load(num_routes_padded_ptr):
        return

    routes = tl.load(sorted_route_ids_ptr + block_m * BLOCK_M +
                     tl.arange(0, BLOCK_M))
    valid_routes = routes < NUM_ROUTES
    expert = tl.load(expert_ids_ptr + block_m).to(tl.int64)
    columns = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, BLOCK_K)
    input_ptrs = input_ptr + (
        (routes.to(tl.int64) // INPUT_TOP_K)[:, None] * stride_im +
        reduction[None, :] * stride_ik)
    # Weight strides also support a transpose view during backward.
    weight_ptrs = weight_ptr + expert * stride_we + (
        reduction[:, None] * stride_wk + columns[None, :] * stride_wn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        valid_k = reduction < K - k * BLOCK_K
        inputs = tl.load(input_ptrs,
                         mask=valid_routes[:, None] & valid_k[None, :],
                         other=0.0)
        weights = tl.load(weight_ptrs,
                          mask=valid_k[:, None] & (columns[None, :] < N),
                          other=0.0)
        # Do not silently round FP32 inputs to TF32. FP16/BF16 still use tensor
        # cores with FP32 accumulation; input_precision is ignored for them.
        accumulator = tl.dot(inputs,
                             weights,
                             acc=accumulator,
                             input_precision="ieee")
        input_ptrs += BLOCK_K * stride_ik
        weight_ptrs += BLOCK_K * stride_wk

    output_ptrs = (output_ptr + routes.to(tl.int64)[:, None] * N +
                   columns[None, :])
    tl.store(output_ptrs,
             accumulator,
             mask=valid_routes[:, None] & (columns[None, :] < N))


def _grouped_linear_forward(inputs: torch.Tensor, weights: torch.Tensor,
                            sorted_route_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_routes_padded: torch.Tensor,
                            input_top_k: int) -> torch.Tensor:
    """Return expert matmuls in original token/slot order, without a gather."""
    num_routes = inputs.shape[0] * input_top_k
    output = inputs.new_empty((num_routes, weights.shape[1]))
    if num_routes:
        grid = (triton.cdiv(sorted_route_ids.numel(), _BLOCK_M) *
                triton.cdiv(weights.shape[1], _BLOCK_N), )
        with torch.cuda.device(inputs.device):
            _grouped_gemm_kernel[grid](
                inputs,
                weights,
                output,
                sorted_route_ids,
                expert_ids,
                num_routes_padded,
                num_routes,
                weights.shape[1],
                inputs.shape[1],
                inputs.stride(0),
                inputs.stride(1),
                weights.stride(0),
                weights.stride(1),
                weights.stride(2),
                input_top_k,
                BLOCK_M=_BLOCK_M,
                BLOCK_N=_BLOCK_N,
                BLOCK_K=_BLOCK_K,
                num_warps=4,
                num_stages=3,
            )
    return output


class _GroupedLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, weights, sorted_route_ids, expert_ids,
                num_routes_padded, input_top_k):
        # No input activations need to be retained: frozen weights only require
        # dInput = dOutput @ W. The route plan is shared by both linear layers.
        ctx.save_for_backward(weights, sorted_route_ids, expert_ids,
                              num_routes_padded)
        ctx.input_top_k = input_top_k
        ctx.input_rows = inputs.shape[0]
        return _grouped_linear_forward(inputs, weights, sorted_route_ids,
                                       expert_ids, num_routes_padded,
                                       input_top_k)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return (None, ) * 6
        weights, sorted_route_ids, expert_ids, num_routes_padded = (
            ctx.saved_tensors)
        grad_input = _grouped_linear_forward(grad_output,
                                             weights.transpose(1, 2),
                                             sorted_route_ids, expert_ids,
                                             num_routes_padded, 1)
        if ctx.input_top_k != 1:
            # The first linear virtually replicated each token over its top-k
            # experts. Sum their gradients without contended atomic writes.
            grad_input = grad_input.view(ctx.input_rows, ctx.input_top_k,
                                        weights.shape[2]).sum(dim=1)
        return grad_input, None, None, None, None, None


def grouped_moe(hidden_states: torch.Tensor, w13: torch.Tensor,
                w2: torch.Tensor, topk_ids: torch.Tensor,
                routing_weights: torch.Tensor) -> torch.Tensor:
    """Apply frozen SiLU experts with grouped GEMMs and first-order gradients.

    Arguments are CUDA tensors on one device. ``hidden_states`` is ``[T, H]``,
    ``w13`` is packed gate/up weights ``[E, 2 * I, H]``, and ``w2`` is
    ``[E, H, I]``. Each TP rank supplies its local intermediate-dimension shard;
    the caller performs the TP output reduction. The floating tensors must have
    the same FP16, BF16 or FP32 dtype. ``topk_ids`` and ``routing_weights`` are
    ``[T, top_k]``. Expert IDs must be in ``[0, E)`` (as produced by topk).

    Routing, including optional normalization, is the caller's responsibility.
    Gradients flow to hidden states and routing weights, never expert weights.
    Higher-order differentiation, quantized experts and expert LoRA are not
    supported. No host reads of CUDA route counts or per-expert launches occur.
    """
    if hidden_states.ndim != 2 or w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("grouped_moe expects hidden states [T, H] and "
                         "expert weights [E, 2 * I, H] / [E, H, I].")
    num_tokens, hidden_size = hidden_states.shape
    num_experts, intermediate_twice, weight_hidden_size = w13.shape
    if (num_experts <= 0 or hidden_size <= 0 or intermediate_twice <= 0
            or intermediate_twice % 2 or weight_hidden_size != hidden_size
            or w2.shape != (num_experts, hidden_size,
                            intermediate_twice // 2)):
        raise ValueError("grouped_moe expert weight shapes do not match "
                         "the hidden and intermediate dimensions.")
    if (topk_ids.ndim != 2 or topk_ids.shape[0] != num_tokens
            or not 0 < topk_ids.shape[1] <= num_experts
            or routing_weights.shape != topk_ids.shape):
        raise ValueError("grouped_moe routing tensors must have matching "
                         "[T, top_k] shapes with 0 < top_k <= E.")
    if w13.requires_grad or w2.requires_grad:
        raise ValueError("grouped_moe requires frozen expert weights; "
                         "expert-weight gradients are not implemented.")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("grouped_moe expert IDs must be int32 or int64.")
    tensors = (hidden_states, w13, w2, topk_ids, routing_weights)
    if (not hidden_states.is_cuda
            or any(t.device != hidden_states.device for t in tensors)):
        raise ValueError("grouped_moe requires all tensors on one CUDA device.")
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if (hidden_states.dtype not in supported_dtypes
            or any(t.dtype != hidden_states.dtype
                   for t in (w13, w2, routing_weights))):
        raise ValueError("grouped_moe requires matching FP16, BF16 or FP32 "
                         "dtypes for inputs, experts and routing weights.")
    if topk_ids.numel() + num_experts * (_BLOCK_M - 1) >= 2**31:
        raise ValueError("grouped_moe route count exceeds int32 indexing.")
    if num_tokens == 0:
        # Retain empty input and routing gradients without expert kernel calls.
        return hidden_states + routing_weights.sum() * 0

    top_k = topk_ids.shape[1]
    with torch.cuda.device(hidden_states.device):
        sorted_route_ids, expert_ids, num_routes_padded = moe_align_block_size(
            topk_ids.contiguous(), _BLOCK_M, num_experts)
    gate_up = _GroupedLinear.apply(hidden_states, w13, sorted_route_ids,
                                   expert_ids, num_routes_padded, top_k)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = F.silu(gate) * up
    expert_output = _GroupedLinear.apply(activated, w2, sorted_route_ids,
                                         expert_ids, num_routes_padded, 1)
    expert_output = expert_output.view(num_tokens, top_k, hidden_size)
    return (expert_output * routing_weights.unsqueeze(-1)).sum(dim=1)
