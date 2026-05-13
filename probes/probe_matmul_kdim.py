"""Find which BLOCK_K sizes are usable in tl.dot on V100."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import torch, triton
import triton.language as tl

print("device:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0))
print("triton:", triton.__version__)


@triton.jit
def _mm(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16))


torch.manual_seed(0)
M = N = K = 256
a = torch.randn((M, K), dtype=torch.float16, device="cuda")
b = torch.randn((K, N), dtype=torch.float16, device="cuda")
ref = (a.float() @ b.float()).half()
for BK in (16, 32, 64, 128, 256):
    c = torch.empty((M, N), dtype=torch.float16, device="cuda")
    grid = (M // 64, N // 64)
    try:
        _mm[grid](a, b, c, M, N, K,
                  a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                  c.stride(0), c.stride(1),
                  BLOCK_M=64, BLOCK_N=64, BLOCK_K=BK)
        torch.cuda.synchronize()
        err = (c.float() - ref.float()).abs().max().item()
        print(f"BLOCK_K={BK:4d}: err={err:.3e}  {'OK' if err < 1.0 else 'FAIL'}")
    except Exception as e:
        print(f"BLOCK_K={BK:4d}: compile FAIL: {type(e).__name__}: {str(e)[:80]}")
