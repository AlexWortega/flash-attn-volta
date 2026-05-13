"""Does num_stages=1 sidestep the Volta layout bug?"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import torch, triton
from triton.ops.flash_attention import _fwd_kernel

print("device:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0))

B, H, N, D = 1, 2, 128, 64
q = torch.randn((B, H, N, D), dtype=torch.float16, device="cuda")
k = torch.randn((B, H, N, D), dtype=torch.float16, device="cuda")
v = torch.randn((B, H, N, D), dtype=torch.float16, device="cuda")
o = torch.empty_like(q)
L = torch.empty((B * H, N), device="cuda", dtype=torch.float32)
m = torch.empty((B * H, N), device="cuda", dtype=torch.float32)

BLOCK = 128
grid = (triton.cdiv(N, BLOCK), B * H, 1)
sm_scale = 1.0 / (D ** 0.5)

try:
    _fwd_kernel[grid](
        q, k, v, sm_scale, L, m, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, N,
        BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_DMODEL=D,
        num_warps=4, num_stages=1,
    )
    torch.cuda.synchronize()
    print("num_stages=1 OK")
except Exception as e:
    print("num_stages=1 FAIL:", type(e).__name__)
