"""Compare Qwen3 LMS grouped GEMMs against the per-expert PyTorch loop.

Uses synthetic inputs and frozen weights; no checkpoint or download is needed.
Defaults model one TP=4 shard of Qwen3-30B-A3B on a single GPU. Timings include
router computation and CPU dispatch overhead, but exclude TP collectives and
the rest of the model. Forward timings retain autograd, as during training.

Example:
    python benchmarks/kernels/benchmark_grouped_moe.py --tp-size 4
"""

import argparse
import time
from typing import Callable

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.grouped_gemm import grouped_moe


def expert_loop(hidden_states: torch.Tensor, w13: torch.Tensor,
                w2: torch.Tensor, topk_ids: torch.Tensor,
                routing_weights: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(hidden_states)
    for expert in range(w13.shape[0]):
        token_ids, slots = torch.where(topk_ids == expert)
        if token_ids.numel() == 0:
            continue
        gate, up = F.linear(hidden_states[token_ids], w13[expert]).chunk(2, -1)
        expert_output = F.linear(F.silu(gate) * up, w2[expert])
        expert_output = expert_output * routing_weights[token_ids, slots, None]
        output.index_add_(0, token_ids, expert_output)
    return output


def measure(run: Callable[[], None], warmup: int, iters: int) -> float:
    # Warmup includes both forward and backward JIT compilation when applicable.
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"],
                        default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--no-renormalize", action="store_true")
    args = parser.parse_args()
    for name in ("tokens", "hidden_size", "intermediate_size", "num_experts",
                 "top_k", "tp_size", "warmup", "iters"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.top_k > args.num_experts:
        parser.error("--top-k cannot exceed --num-experts")
    if args.intermediate_size % args.tp_size:
        parser.error("--intermediate-size must be divisible by --tp-size")
    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required")
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("the selected CUDA GPU does not support BF16")

    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[args.dtype]
    hidden = args.hidden_size
    intermediate = args.intermediate_size // args.tp_size
    inputs = torch.randn(args.tokens, hidden, device="cuda", dtype=dtype,
                         requires_grad=True)
    w13 = torch.randn(args.num_experts, 2 * intermediate, hidden,
                      device="cuda", dtype=dtype) / hidden**0.5
    w2 = torch.randn(args.num_experts, hidden, intermediate,
                     device="cuda", dtype=dtype) / intermediate**0.5
    router = torch.randn(args.num_experts, hidden, device="cuda",
                         dtype=dtype) / hidden**0.5
    grad_output = torch.randn_like(inputs)

    print(f"GPU: {torch.cuda.get_device_name()}; dtype: {args.dtype}")
    print(f"tokens={args.tokens}, hidden={hidden}, experts={args.num_experts}, "
          f"top_k={args.top_k}, renormalize={not args.no_renormalize}")
    print(f"One TP={args.tp_size} shard: intermediate={intermediate}; "
          "TP collectives excluded. Forward includes autograd graph creation.")
    print(f"Warmup={args.warmup}, measured iterations={args.iters}")

    for backward in (False, True):
        timings = {}
        for name, implementation in (("expert_loop", expert_loop),
                                     ("grouped_gemm", grouped_moe)):

            def run(implementation=implementation, backward=backward) -> None:
                inputs.grad = None
                logits = F.linear(inputs, router)
                probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
                weights, ids = probabilities.topk(args.top_k, dim=-1)
                if not args.no_renormalize:
                    weights = weights / weights.sum(dim=-1, keepdim=True)
                output = implementation(inputs, w13, w2, ids,
                                        weights.to(dtype))
                if backward:
                    output.backward(grad_output)

            timings[name] = measure(run, args.warmup, args.iters)
        mode = "forward+backward" if backward else "forward"
        baseline, grouped = timings["expert_loop"], timings["grouped_gemm"]
        print(f"{mode}: expert_loop={baseline:.3f} ms, "
              f"grouped_gemm={grouped:.3f} ms, ratio={baseline / grouped:.2f}x")


if __name__ == "__main__":
    main()
