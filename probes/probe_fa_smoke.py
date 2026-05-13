"""Smoke test — does flash_attn_forward compile + run end-to-end?"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from flash_attn_volta import flash_attn_forward
from flash_attn_volta.ref import attention_ref

torch.manual_seed(0)
device = "cuda"
print("device:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0))

for B, N, H, D in [(1, 64, 2, 32), (2, 128, 4, 64), (1, 256, 2, 128)]:
    for causal in (False, True):
        q = torch.randn((B, N, H, D), dtype=torch.float16, device=device)
        k = torch.randn((B, N, H, D), dtype=torch.float16, device=device)
        v = torch.randn((B, N, H, D), dtype=torch.float16, device=device)
        out = flash_attn_forward(q, k, v, causal=causal)
        ref = attention_ref(q, k, v, causal=causal)
        err = (out.float() - ref.float()).abs().max().item()
        print(f"shape=({B},{N},{H},{D}) causal={causal}  out.shape={tuple(out.shape)}  max_err={err:.4e}")
print("DONE")
