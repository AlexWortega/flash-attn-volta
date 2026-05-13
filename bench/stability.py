"""Large-seq numerical stability: confirm no NaN/Inf at seq>=4096."""
from __future__ import annotations
import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from flash_attn_volta import flash_attn_forward


def main():
    torch.manual_seed(0)
    print("device:", torch.cuda.get_device_name(0),
          "cap:", torch.cuda.get_device_capability(0))
    results = []
    overall_pass = True
    for (B, N, H, D), causal in [
        ((1, 4096, 16, 64), False),
        ((1, 4096, 16, 64), True),
        ((1, 4096, 8, 128), False),
        ((1, 8192, 4, 64), False),
        ((1, 8192, 4, 64), True),
    ]:
        q = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda") * 0.5
        k = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda") * 0.5
        v = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda") * 0.5
        out = flash_attn_forward(q, k, v, causal=causal)
        n_nan = torch.isnan(out).sum().item()
        n_inf = torch.isinf(out).sum().item()
        out_min = out.float().min().item()
        out_max = out.float().max().item()
        verdict = "pass" if (n_nan == 0 and n_inf == 0) else "FAIL"
        overall_pass &= (verdict == "pass")
        line = (f"shape=({B},{N},{H},{D}) causal={str(causal):5s}  "
                f"nan={n_nan} inf={n_inf}  range=[{out_min:.3e},{out_max:.3e}]  "
                f"{verdict}")
        print(line)
        results.append({
            "B": B, "N": N, "H": H, "D": D, "causal": causal,
            "n_nan": n_nan, "n_inf": n_inf,
            "out_min": out_min, "out_max": out_max,
            "verdict": verdict,
        })
    print()
    print("STABILITY:", "PASS" if overall_pass else "FAIL")
    out_path = os.path.join(os.path.dirname(__file__), "..",
                            "results", "stability.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": results, "overall_pass": overall_pass}, f, indent=2)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
