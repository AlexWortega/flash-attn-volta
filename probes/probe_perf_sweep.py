"""Sweep (BLOCK_M, BLOCK_N, num_warps, num_stages) for D=64 perf on seq=1024."""
import os, sys, time, math
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, triton
from flash_attn_volta import triton_fa as fa_mod
from flash_attn_volta.ref import attention_ref

torch.manual_seed(0)
B, N, H, D = 1, 1024, 16, 64
q = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda")
k = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda")
v = torch.randn((B, N, H, D), dtype=torch.float16, device="cuda")
ref = attention_ref(q, k, v, causal=False).float()
flops_fn = lambda: 4.0 * B * H * N * N * D


def time_fn(fn, *args, warmup=5, iters=30):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


print("device:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0))
print("triton:", triton.__version__)

best = None
for BM in (64, 128):
    for BN in (32, 64, 128):
        for nw in (2, 4, 8):
            for ns in (1, 2, 3):
                if BM * BN > 16384:
                    continue
                fa_mod._pick_block_sizes = lambda dim, _BM=BM, _BN=BN, _nw=nw, _ns=ns: (_BM, _BN, _nw, _ns)
                try:
                    out = fa_mod.flash_attn_forward(q, k, v, causal=False).float()
                    err = (out - ref).abs().max().item()
                    if err > 1e-2:
                        continue
                    t = time_fn(fa_mod.flash_attn_forward, q, k, v, False)
                    tf = flops_fn() / t / 1e12
                    msg = f"BM={BM:3d} BN={BN:3d} nw={nw} ns={ns}: t={t*1e3:.3f} ms  tflops={tf:6.2f}"
                    print(msg)
                    if best is None or tf > best[0]:
                        best = (tf, (BM, BN, nw, ns))
                except Exception as e:
                    pass  # silent skip
print()
print("BEST:", best)
