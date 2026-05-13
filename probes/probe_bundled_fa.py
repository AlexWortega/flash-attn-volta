"""Run Triton's bundled FA tutorial kernel directly (bypassing the cap-8 gate)."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import torch
import triton
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

_fwd_kernel[grid](
    q, k, v, sm_scale,
    L, m,
    o,
    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
    o.stride(0), o.stride(1), o.stride(2), o.stride(3),
    B, H, N,
    BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_DMODEL=D,
    num_warps=4, num_stages=2,
)
torch.cuda.synchronize()

import math
q_f = q.float(); k_f = k.float(); v_f = v.float()
scores = torch.matmul(q_f, k_f.transpose(-1, -2)) * sm_scale
mask = torch.triu(torch.full((N, N), float("-inf"), device="cuda"), diagonal=1)
scores = scores + mask
probs = torch.softmax(scores, dim=-1)
ref = torch.matmul(probs, v_f)
err = (o.float() - ref).abs().max().item()
print("bundled FA max-abs err (causal):", err)
