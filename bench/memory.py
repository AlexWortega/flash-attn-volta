"""Peak GPU memory for our kernel vs torch eager at seq=4096, h=16, d=64."""
from __future__ import annotations
import os, sys, json, gc, math
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from flash_attn_volta import flash_attn_forward


def _bytes(t):
    return t.numel() * t.element_size()


def _peak_for(fn, q, k, v, causal):
    """Measure peak allocator memory while fn runs."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn(q, k, v, causal=causal)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return base, peak - base


def _torch_eager(q, k, v, causal):
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
    return out


def main():
    torch.manual_seed(0)
    print("device:", torch.cuda.get_device_name(0),
          "cap:", torch.cuda.get_device_capability(0))
    rows = []

    for (B, N, H, D), causal in [
        ((1, 1024, 16, 64), False),
        ((1, 2048, 16, 64), False),
        ((1, 4096, 16, 64), False),
        ((1, 4096, 16, 64), True),
    ]:
        gc.collect()
        torch.cuda.empty_cache()
        q = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda")
        k = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda")
        v = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda")

        # FA
        base_fa, extra_fa = _peak_for(
            lambda q, k, v, causal: flash_attn_forward(q, k, v, causal=causal),
            q, k, v, causal,
        )
        # Eager (can OOM at large N)
        try:
            gc.collect(); torch.cuda.empty_cache()
            base_eg, extra_eg = _peak_for(_torch_eager, q, k, v, causal)
        except torch.cuda.OutOfMemoryError:
            base_eg, extra_eg = -1, -1

        qkv_bytes = _bytes(q) + _bytes(k) + _bytes(v)
        # Theoretical FA extra memory: just output (b*h*n*d*2 bytes), no n*n matrix.
        # Theoretical eager extra memory: scores n*n (in fp16) per (b,h) ~ b*h*n*n*2.
        theory_fa_extra = B * N * H * D * 2  # output
        theory_eg_extra = B * H * N * N * 2  # scores

        rows.append({
            "B": B, "N": N, "H": H, "D": D, "causal": causal,
            "qkv_bytes": qkv_bytes,
            "fa_extra_bytes": extra_fa,
            "eager_extra_bytes": extra_eg,
            "theory_fa_extra_bytes": theory_fa_extra,
            "theory_eager_extra_bytes": theory_eg_extra,
        })

        print(f"shape=({B},{N},{H},{D}) causal={causal}  "
              f"  fa_extra={extra_fa/1e6:8.2f} MB  "
              f"  eager_extra={('OOM' if extra_eg < 0 else f'{extra_eg/1e6:8.2f} MB')}  "
              f"  theory_fa={theory_fa_extra/1e6:.2f} MB  "
              f"  theory_eager={theory_eg_extra/1e6:.2f} MB")

        del q, k, v
        gc.collect(); torch.cuda.empty_cache()

    out_path = os.path.join(os.path.dirname(__file__), "..",
                            "results", "memory.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
