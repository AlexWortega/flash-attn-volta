"""Isolate the D=32 failure: try different (BLOCK_M, BLOCK_N) tile sizes."""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import math, torch, triton
import triton.language as tl
from flash_attn_volta.ref import attention_ref
from flash_attn_volta import triton_fa as fa_mod


torch.manual_seed(0)
print("device:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0))

# tiny case — small N, D=32
B, N, H, D = 1, 64, 2, 32
q = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda")
k = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda")
v = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda")
ref = attention_ref(q, k, v, causal=False).float()


def run(BLOCK_M, BLOCK_N, num_warps):
    fa_mod._pick_block_sizes = lambda dim: (BLOCK_M, BLOCK_N, num_warps, 1)
    out = fa_mod.flash_attn_forward(q, k, v, causal=False).float()
    err = (out - ref).abs().max().item()
    return err


for BM, BN, nw in [(64, 64, 4), (32, 32, 4), (64, 32, 4), (128, 64, 4), (32, 64, 4), (128, 32, 4)]:
    try:
        e = run(BM, BN, nw)
        print(f"BLOCK_M={BM:3d} BLOCK_N={BN:3d} nw={nw}: err={e:.3e}  {'ok' if e < 1e-2 else 'FAIL'}")
    except Exception as e:
        print(f"BLOCK_M={BM:3d} BLOCK_N={BN:3d} nw={nw}: {type(e).__name__}: {str(e)[:90]}")
