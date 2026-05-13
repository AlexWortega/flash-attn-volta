"""Correctness suite for flash_attn_volta.

For every required (shape, causal) combination compute:
    out_fa   = flash_attn_forward(q, k, v, causal=...)
    out_sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=...)
    err = (out_fa - out_sdpa).abs().max()

Pass if err < 1e-2 (the brief's tolerance, fp16-friendly).

We also report errors against an fp32 reference (`attention_ref`) as a sanity
diagnostic -- not part of the pass/fail (SDPA-fp16 reference is what the brief
asks for).
"""
from __future__ import annotations
import os, sys, math, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from flash_attn_volta import flash_attn_forward
from flash_attn_volta.ref import attention_ref


SHAPES = [
    (2, 1024, 8, 64),
    (1, 2048, 16, 128),
    (4, 512, 4, 32),
]
TOL = 1.0e-2  # 1e-2 max-abs fp16


def _sdpa_ref(q, k, v, causal):
    """Match torch's expected layout: (B, H, N, D)."""
    q_ = q.permute(0, 2, 1, 3).contiguous()
    k_ = k.permute(0, 2, 1, 3).contiguous()
    v_ = v.permute(0, 2, 1, 3).contiguous()
    out = F.scaled_dot_product_attention(q_, k_, v_, is_causal=causal)
    return out.permute(0, 2, 1, 3).contiguous()


def main():
    print("device:", torch.cuda.get_device_name(0),
          "cap:", torch.cuda.get_device_capability(0))
    torch.manual_seed(0)
    results = []
    overall_pass = True

    for B, N, H, D in SHAPES:
        for causal in (False, True):
            q = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda")
            k = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda")
            v = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda")

            out_fa = flash_attn_forward(q, k, v, causal=causal)
            out_sdpa = _sdpa_ref(q, k, v, causal)
            out_fp32 = attention_ref(q, k, v, causal=causal)

            err_sdpa = (out_fa.float() - out_sdpa.float()).abs().max().item()
            err_fp32 = (out_fa.float() - out_fp32.float()).abs().max().item()
            verdict = "pass" if err_sdpa < TOL else "FAIL"
            overall_pass &= (verdict == "pass")

            line = (f"shape=({B},{N},{H},{D}) causal={str(causal):5s}  "
                    f"err_vs_sdpa={err_sdpa:.3e}  err_vs_fp32={err_fp32:.3e}  "
                    f"{verdict}")
            print(line)
            results.append({
                "B": B, "N": N, "H": H, "D": D, "causal": causal,
                "err_vs_sdpa": err_sdpa, "err_vs_fp32": err_fp32,
                "verdict": verdict,
            })

    print()
    print("OVERALL:", "PASS" if overall_pass else "FAIL")
    out_path = os.path.join(os.path.dirname(__file__), "..",
                            "results", "correctness.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": results, "overall_pass": overall_pass,
                   "tol": TOL}, f, indent=2)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
