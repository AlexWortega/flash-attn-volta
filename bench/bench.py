"""Throughput benchmark.

Compares flash_attn_volta vs:
  (a) torch eager attention:        (q @ k.T / sqrt(d)).softmax(-1) @ v
  (b) torch sdpa math backend:      F.scaled_dot_product_attention with FA disabled

Forward FLOPs counted as 4 * B * H * S * S * D.

For causal we still report 4 * B * H * S * S * D as the "dense" FLOP count
(common convention -- gives a comparable speedup number against dense torch).

Outputs a small JSON next to this script and prints a table.
"""
from __future__ import annotations
import os, sys, json, math, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from flash_attn_volta import flash_attn_forward


def _torch_eager(q, k, v, causal):
    """Plain (q @ k.T / sqrt(d)).softmax(-1) @ v in fp16."""
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


def _sdpa_math(q, k, v, causal):
    """torch sdpa, forced to the slow 'math' backend (FA / mem-efficient disabled).
    Layout: (B, H, N, D).
    """
    q_ = q.permute(0, 2, 1, 3).contiguous()
    k_ = k.permute(0, 2, 1, 3).contiguous()
    v_ = v.permute(0, 2, 1, 3).contiguous()
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=True, enable_mem_efficient=False
    ):
        out = F.scaled_dot_product_attention(q_, k_, v_, is_causal=causal)
    return out.permute(0, 2, 1, 3).contiguous()


def _time_one(fn, *args, warmup=5, iters=20):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters, out


def _flops_fwd(B, N, H, D):
    """Forward FLOPs for full (non-causal) attention. 4 * B * H * S * S * D."""
    return 4.0 * B * H * N * N * D


def bench_shape(B, N, H, D, causal, iters=20):
    q = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda")
    k = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda")
    v = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda")

    # FA
    t_fa, _ = _time_one(flash_attn_forward, q, k, v, causal, iters=iters)
    # Torch eager
    try:
        t_eager, _ = _time_one(_torch_eager, q, k, v, causal, iters=iters)
    except torch.cuda.OutOfMemoryError:
        t_eager = float("nan")
    # SDPA math
    try:
        t_sdpa, _ = _time_one(_sdpa_math, q, k, v, causal, iters=iters)
    except torch.cuda.OutOfMemoryError:
        t_sdpa = float("nan")

    flops = _flops_fwd(B, N, H, D)
    return {
        "B": B, "N": N, "H": H, "D": D, "causal": causal,
        "t_fa_s": t_fa, "t_eager_s": t_eager, "t_sdpa_math_s": t_sdpa,
        "tflops_fa": flops / t_fa / 1e12,
        "tflops_eager": flops / t_eager / 1e12 if t_eager == t_eager else float("nan"),
        "tflops_sdpa_math": flops / t_sdpa / 1e12 if t_sdpa == t_sdpa else float("nan"),
        "speedup_vs_eager": t_eager / t_fa if t_eager == t_eager else float("nan"),
        "speedup_vs_sdpa": t_sdpa / t_fa if t_sdpa == t_sdpa else float("nan"),
    }


def main():
    print("device:", torch.cuda.get_device_name(0),
          "cap:", torch.cuda.get_device_capability(0))
    shapes = [
        (1, 1024, 16, 64, False),
        (1, 1024, 16, 64, True),
        (1, 2048, 16, 64, False),
        (1, 2048, 16, 64, True),
        (1, 4096, 16, 64, False),
        (1, 4096, 8, 128, False),
        (1, 1024, 8, 128, False),
    ]
    rows = []
    for B, N, H, D, causal in shapes:
        r = bench_shape(B, N, H, D, causal)
        rows.append(r)
        print(f"  ({B},{N},{H},{D}) causal={str(causal):5s} "
              f" fa={r['tflops_fa']:6.2f} TFLOP/s "
              f" eager={r['tflops_eager']:6.2f} "
              f" sdpa={r['tflops_sdpa_math']:6.2f} "
              f" speedup(eager)={r['speedup_vs_eager']:.2f}x "
              f" speedup(sdpa)={r['speedup_vs_sdpa']:.2f}x")

    out_path = os.path.join(os.path.dirname(__file__), "..",
                            "results", "bench.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
