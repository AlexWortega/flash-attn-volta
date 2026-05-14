"""Forward + backward throughput / memory benchmark.

Compares `flash_attn` (autograd-aware) vs torch eager attention, reporting:
  - Combined fwd+bwd TFLOP/s. fwd FLOPs = 4·B·H·S²·D, bwd FLOPs ≈ 4× that
    per the FlashAttention paper (Dao et al. §B). So total = 5·fwd_flops.
  - Peak GPU memory for fwd+bwd at seq 1k/2k/4k. The eager path materialises
    the O(S²) attention matrix for backward; flash_attn does not.

For each shape we run causal=False and causal=True. We always count the same
"dense" FLOP number against the wall-clock — it lets the speedup column be
read directly as "flash_attn fwd+bwd vs eager fwd+bwd" without having to
discount causal at the expense of comparability.
"""
from __future__ import annotations
import os, sys, json, math, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from flash_attn_volta import flash_attn


def _torch_eager(q, k, v, causal):
    """Plain (q @ k.T / sqrt(d)).softmax(-1) @ v. Differentiable."""
    B, N, H, D = q.shape
    q_ = q.permute(0, 2, 1, 3)
    k_ = k.permute(0, 2, 1, 3)
    v_ = v.permute(0, 2, 1, 3)
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q_, k_.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(
            torch.full((N, N), float("-inf"), device=q.device, dtype=scores.dtype),
            diagonal=1,
        )
        scores = scores + mask
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_)
    return out.permute(0, 2, 1, 3).contiguous()


def _fwd_bwd_step(fn, q, k, v, do, causal):
    """One forward+backward step. Returns wall time and grads-allocated bool."""
    q.grad = None; k.grad = None; v.grad = None
    out = fn(q, k, v, causal)
    out.backward(do)
    return q.grad, k.grad, v.grad


def _time_fwd_bwd(fn, B, N, H, D, causal, warmup=3, iters=10):
    q = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda", requires_grad=True)
    k = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda", requires_grad=True)
    v = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda", requires_grad=True)
    do = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda")

    # warmup
    for _ in range(warmup):
        _fwd_bwd_step(fn, q, k, v, do, causal)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _fwd_bwd_step(fn, q, k, v, do, causal)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def _peak_mem_fwd_bwd(fn, B, N, H, D, causal):
    """Peak allocator memory across forward+backward of one step."""
    q = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda", requires_grad=True)
    k = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda", requires_grad=True)
    v = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda", requires_grad=True)
    do = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda")
    # Warm + reset
    _fwd_bwd_step(fn, q, k, v, do, causal)
    q.grad = None; k.grad = None; v.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn(q, k, v, causal)
    out.backward(do)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    return peak


def bench_throughput(B, N, H, D, causal, iters=10):
    flops_fwd = 4.0 * B * H * N * N * D
    # FA paper §B: bwd ≈ 4× fwd. Total = 5 × fwd.
    flops_total = 5.0 * flops_fwd

    # FA path
    def fa_fn(q, k, v, c):
        return flash_attn(q, k, v, causal=c)
    t_fa = _time_fwd_bwd(fa_fn, B, N, H, D, causal, iters=iters)

    # Eager path
    def eager_fn(q, k, v, c):
        return _torch_eager(q, k, v, c)
    try:
        t_eager = _time_fwd_bwd(eager_fn, B, N, H, D, causal, iters=iters)
    except torch.cuda.OutOfMemoryError:
        t_eager = float("nan")

    return {
        "B": B, "N": N, "H": H, "D": D, "causal": causal,
        "t_fa_s":    t_fa,    "tflops_fa":    flops_total / t_fa / 1e12,
        "t_eager_s": t_eager, "tflops_eager": flops_total / t_eager / 1e12 if t_eager == t_eager else float("nan"),
        "speedup_vs_eager": t_eager / t_fa if t_eager == t_eager else float("nan"),
    }


def bench_memory(B, N, H, D, causal):
    def fa_fn(q, k, v, c):
        return flash_attn(q, k, v, causal=c)
    def eager_fn(q, k, v, c):
        return _torch_eager(q, k, v, c)
    fa_mem = _peak_mem_fwd_bwd(fa_fn, B, N, H, D, causal)
    try:
        eager_mem = _peak_mem_fwd_bwd(eager_fn, B, N, H, D, causal)
    except torch.cuda.OutOfMemoryError:
        eager_mem = float("nan")
    return {
        "B": B, "N": N, "H": H, "D": D, "causal": causal,
        "fa_peak_extra_mb": fa_mem / 1024 / 1024,
        "eager_peak_extra_mb": eager_mem / 1024 / 1024 if eager_mem == eager_mem else float("nan"),
    }


def main():
    print("device:", torch.cuda.get_device_name(0),
          "cap:", torch.cuda.get_device_capability(0))

    print("\n=== fwd+bwd throughput ===")
    print(f"{'shape':<22} {'causal':<6} {'fa TF/s':>10} {'eager TF/s':>12} {'speedup':>10}")
    thr_rows = []
    for B, N, H, D, causal in [
        (1, 1024, 16, 64, False),
        (1, 1024, 16, 64, True),
        (1, 2048, 16, 64, False),
        (1, 2048, 16, 64, True),
        (1, 4096, 16, 64, False),
        (1, 4096, 16, 64, True),
        (1, 1024,  8, 128, False),
        (1, 2048,  8, 128, False),
    ]:
        r = bench_throughput(B, N, H, D, causal)
        thr_rows.append(r)
        eager_str = f"{r['tflops_eager']:.2f}" if r['tflops_eager'] == r['tflops_eager'] else "OOM"
        sp_str = f"{r['speedup_vs_eager']:.2f}x" if r['speedup_vs_eager'] == r['speedup_vs_eager'] else "n/a"
        print(f"({B},{N},{H},{D})    {str(causal):<5} "
              f"{r['tflops_fa']:>10.2f} {eager_str:>12s} {sp_str:>10s}")

    print("\n=== peak memory (fwd+bwd, extra over Q+K+V resident) ===")
    print(f"{'shape':<22} {'causal':<6} {'fa MB':>10} {'eager MB':>12} {'ratio':>10}")
    mem_rows = []
    for B, N, H, D, causal in [
        (1, 1024, 16, 64, False),
        (1, 2048, 16, 64, False),
        (1, 4096, 16, 64, False),
        (1, 1024, 16, 64, True),
        (1, 2048, 16, 64, True),
        (1, 4096, 16, 64, True),
    ]:
        r = bench_memory(B, N, H, D, causal)
        mem_rows.append(r)
        eager_str = f"{r['eager_peak_extra_mb']:.1f}" if r['eager_peak_extra_mb'] == r['eager_peak_extra_mb'] else "OOM"
        ratio = (r['eager_peak_extra_mb'] / r['fa_peak_extra_mb']) if r['eager_peak_extra_mb'] == r['eager_peak_extra_mb'] else float("nan")
        ratio_str = f"{ratio:.1f}x" if ratio == ratio else "n/a"
        print(f"({B},{N},{H},{D})    {str(causal):<5} "
              f"{r['fa_peak_extra_mb']:>10.1f} {eager_str:>12s} {ratio_str:>10s}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "backward_bench.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"throughput": thr_rows, "memory": mem_rows}, f, indent=2)
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()
