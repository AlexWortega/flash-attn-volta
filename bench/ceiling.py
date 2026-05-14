"""Ceiling benchmark: flash-attn-volta vs xformers Cutlass on V100.

xformers.ops.memory_efficient_attention with the Cutlass backend is the only
fmha op that supports SM 7.0 (Flash needs SM 8.0). Cutlass FwOp is therefore
the realistic perf ceiling on V100. We benchmark fwd and fwd+bwd on the same
shapes as bench/bench.py and bench/backward.py.

FLOP counting matches our other benches: 4*B*H*S*S*D for forward, 5x for fwd+bwd.
"""
from __future__ import annotations
import os, sys, json, math, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import xformers.ops as xops
from xformers.ops import LowerTriangularMask

from flash_attn_volta import flash_attn_forward, flash_attn


CUTLASS = xops.MemoryEfficientAttentionCutlassOp


def _xf_fwd(q, k, v, causal):
    """xformers Cutlass forward. q/k/v are (B,N,H,D) fp16."""
    bias = LowerTriangularMask() if causal else None
    return xops.memory_efficient_attention_forward(q, k, v, attn_bias=bias, op=CUTLASS[0])


def _xf_fwdbwd(q, k, v, do, causal):
    """xformers Cutlass forward+backward. Requires requires_grad on inputs."""
    q.grad = None; k.grad = None; v.grad = None
    bias = LowerTriangularMask() if causal else None
    o = xops.memory_efficient_attention(q, k, v, attn_bias=bias, op=CUTLASS)
    o.backward(do)
    return q.grad, k.grad, v.grad


def _time(fn, *args, warmup=5, iters=20):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bench_fwd(B, N, H, D, causal, iters=20):
    q = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda")
    k = torch.randn_like(q); v = torch.randn_like(q)
    flops = 4.0 * B * H * N * N * D

    t_fa = _time(flash_attn_forward, q, k, v, causal, iters=iters)
    try:
        t_xf = _time(_xf_fwd, q, k, v, causal, iters=iters)
    except Exception as e:
        return {"B": B, "N": N, "H": H, "D": D, "causal": causal,
                "t_fa_s": t_fa, "t_xf_s": float("nan"),
                "tflops_fa": flops / t_fa / 1e12, "tflops_xf": float("nan"),
                "frac_of_ceiling": float("nan"),
                "xf_error": str(e)[:200]}
    return {"B": B, "N": N, "H": H, "D": D, "causal": causal,
            "t_fa_s": t_fa, "t_xf_s": t_xf,
            "tflops_fa": flops / t_fa / 1e12,
            "tflops_xf": flops / t_xf / 1e12,
            "frac_of_ceiling": t_xf / t_fa}


def bench_fwdbwd(B, N, H, D, causal, iters=10):
    q = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda", requires_grad=True)
    k = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda", requires_grad=True)
    v = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda", requires_grad=True)
    do = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda")
    flops = 5.0 * 4.0 * B * H * N * N * D

    def fa_fn():
        q.grad = None; k.grad = None; v.grad = None
        o = flash_attn(q, k, v, causal=causal)
        o.backward(do)
    def xf_fn():
        _xf_fwdbwd(q, k, v, do, causal)

    t_fa = _time(fa_fn, iters=iters)
    try:
        t_xf = _time(xf_fn, iters=iters)
    except Exception as e:
        return {"B": B, "N": N, "H": H, "D": D, "causal": causal,
                "t_fa_s": t_fa, "t_xf_s": float("nan"),
                "tflops_fa": flops / t_fa / 1e12, "tflops_xf": float("nan"),
                "frac_of_ceiling": float("nan"),
                "xf_error": str(e)[:200]}
    return {"B": B, "N": N, "H": H, "D": D, "causal": causal,
            "t_fa_s": t_fa, "t_xf_s": t_xf,
            "tflops_fa": flops / t_fa / 1e12,
            "tflops_xf": flops / t_xf / 1e12,
            "frac_of_ceiling": t_xf / t_fa}


SHAPES_FWD = [
    (1, 1024, 16, 64,  False),
    (1, 1024, 16, 64,  True),
    (1, 2048, 16, 64,  False),
    (1, 2048, 16, 64,  True),
    (1, 4096, 16, 64,  False),
    (1, 4096, 16, 64,  True),
    (1, 1024,  8, 128, False),
    (1, 2048,  8, 128, False),
    (1, 4096,  8, 128, False),
]

SHAPES_FWDBWD = [
    (1, 1024, 16, 64, False),
    (1, 1024, 16, 64, True),
    (1, 2048, 16, 64, False),
    (1, 2048, 16, 64, True),
    (1, 4096, 16, 64, False),
    (1, 1024,  8, 128, False),
    (1, 2048,  8, 128, False),
]


def main():
    print("device:", torch.cuda.get_device_name(0),
          "cap:", torch.cuda.get_device_capability(0))
    import xformers
    print("xformers:", xformers.__version__)

    print("\n=== forward ===")
    print(f"{'shape':<24} {'causal':<6} {'fa TF/s':>10} {'xf TF/s':>10} {'fa/xf':>8}")
    fwd_rows = []
    for B, N, H, D, c in SHAPES_FWD:
        r = bench_fwd(B, N, H, D, c)
        fwd_rows.append(r)
        xf_s = f"{r['tflops_xf']:.2f}" if r['tflops_xf'] == r['tflops_xf'] else "FAIL"
        ratio = f"{1.0/r['frac_of_ceiling']:.2f}x" if r['frac_of_ceiling'] == r['frac_of_ceiling'] else "n/a"
        print(f"({B},{N},{H},{D})         {str(c):<5} "
              f"{r['tflops_fa']:>10.2f} {xf_s:>10s} {ratio:>8s}")

    print("\n=== forward + backward ===")
    print(f"{'shape':<24} {'causal':<6} {'fa TF/s':>10} {'xf TF/s':>10} {'fa/xf':>8}")
    fwdbwd_rows = []
    for B, N, H, D, c in SHAPES_FWDBWD:
        r = bench_fwdbwd(B, N, H, D, c)
        fwdbwd_rows.append(r)
        xf_s = f"{r['tflops_xf']:.2f}" if r['tflops_xf'] == r['tflops_xf'] else "FAIL"
        ratio = f"{1.0/r['frac_of_ceiling']:.2f}x" if r['frac_of_ceiling'] == r['frac_of_ceiling'] else "n/a"
        print(f"({B},{N},{H},{D})         {str(c):<5} "
              f"{r['tflops_fa']:>10.2f} {xf_s:>10s} {ratio:>8s}")

    out = os.path.join(os.path.dirname(__file__), "..", "results", "ceiling.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"forward": fwd_rows, "fwd_bwd": fwdbwd_rows}, f, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
