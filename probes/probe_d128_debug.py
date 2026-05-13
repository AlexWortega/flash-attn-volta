"""Find a (BLOCK_M, BLOCK_N, num_warps) for D=128 that compiles + is correct."""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import math, torch
from flash_attn_volta.ref import attention_ref
from flash_attn_volta import triton_fa as fa_mod

torch.manual_seed(0)
print("device:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0))

B, N, H, D = 1, 128, 2, 128
q = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda")
k = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda")
v = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda")
ref = attention_ref(q, k, v, causal=False).float()


def run(BM, BN, nw, ns):
    fa_mod._pick_block_sizes = lambda dim: (BM, BN, nw, ns)
    out = fa_mod.flash_attn_forward(q, k, v, causal=False).float()
    return (out - ref).abs().max().item()


configs = [
    (64, 64, 4, 1), (64, 64, 8, 1), (64, 32, 4, 1), (64, 32, 8, 1),
    (32, 64, 4, 1), (32, 32, 4, 1), (32, 64, 8, 1),
    (128, 32, 4, 1), (128, 64, 4, 1), (128, 64, 8, 1),
    (64, 64, 4, 2), (64, 64, 8, 2),
]
for BM, BN, nw, ns in configs:
    try:
        e = run(BM, BN, nw, ns)
        print(f"BM={BM:3d} BN={BN:3d} nw={nw} ns={ns}: err={e:.3e}  {'ok' if e < 1e-2 else 'FAIL'}")
    except Exception as e:
        print(f"BM={BM:3d} BN={BN:3d} nw={nw} ns={ns}: {type(e).__name__}: {str(e)[:80]}")
