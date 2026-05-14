"""Triton FlashAttention forward kernel for NVIDIA Volta (SM 7.0 / V100).

Single-file Triton port of the FlashAttention-1 forward algorithm
(Dao et al., 2022). Tiled QK matmul + online softmax with FP32 accumulation.

External API:
    flash_attn_forward(q, k, v, causal=False, sm_scale=None) -> out

Shapes:
    q, k, v: (B, N, H, D)   fp16   (D in {16, 32, 64, 128})
    out:     (B, N, H, D)   fp16

Volta / Triton-2.x quirks worked around:
    * The Triton-2.3 V100 backend mis-handles `tl.dot` for `head_dim != 64`.
      We therefore:
        - pad `D < 64` to 64 in the Python wrapper (fast, near-zero overhead),
        - split `D == 128` into two BLOCK_D=64 halves inside the kernel, doing
          two `tl.dot`s and two accumulators.
      That leaves every `tl.dot` invocation with K-dim == 64, which is the
      tested-working configuration on V100.
    * `qk = tl.zeros([..], fp32) + tl.dot(q, k)` forces the post-dot tensor
      into a regular blocked layout (avoids the older 2.0 `tt.reduce` mma
      layout-mismatch issue and keeps us safe across Triton versions).
"""

from __future__ import annotations
import math

import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Autotune config sets
# -----------------------------------------------------------------------------

# V100 + Triton 2.3 quirks (empirically discovered, see PERF.md):
#   * tl.dot K-dim must be 64 → BLOCK_D forced to 64.
#   * BLOCK_N ∈ {32, 128} triggers an LLVM-IR codegen IndexError → fix BN=64.
#   * num_warps=8 produces silently-wrong output → keep num_warps=4.
# That leaves BLOCK_M and num_stages as the only safe knobs.

_FWD_CONFIGS_D64 = [
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=3),
]

# D=128 doubles the per-program SMEM/register pressure (two BLOCK_D=64 halves).
# BLOCK_M=128 with ns=3 spilled badly in probes; restrict to BM=64.
_FWD_CONFIGS_D128 = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=3),
]


# -----------------------------------------------------------------------------
# Triton kernel (D = 64, single half)
# -----------------------------------------------------------------------------

@triton.autotune(configs=_FWD_CONFIGS_D64, key=['N_CTX'])
@triton.jit
def _fa_fwd_kernel_d64(
    Q, K, V, Out, Lse,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_lz, stride_lh,
    Z, H, N_CTX,
    IS_CAUSAL: tl.constexpr,
    WRITE_LSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program = one (Q tile, batch_head). Streams K/V tiles. D fixed at BLOCK_D."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + off_z * stride_qz + off_h * stride_qh \
        + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + off_z * stride_kz + off_h * stride_kh \
        + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_ptrs = V + off_z * stride_vz + off_h * stride_vh \
        + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    out_ptrs = Out + off_z * stride_oz + off_h * stride_oh \
        + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on

    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Load Q -- masked load when N may not be a multiple of BLOCK_M (always true
    # for the row dimension via the q_row_mask check below).
    q_row_mask = offs_m < N_CTX
    q = tl.load(q_ptrs, mask=q_row_mask[:, None], other=0.0)
    # Pre-scale Q once so we don't multiply qk by sm_scale every iteration.
    q = (q.to(tl.float32) * sm_scale).to(q.dtype)

    if IS_CAUSAL:
        hi = (start_m + 1) * BLOCK_M
        if hi > N_CTX:
            hi = N_CTX
    else:
        hi = N_CTX

    for start_n in range(0, hi, BLOCK_N):
        n_col_mask = (start_n + offs_n) < N_CTX
        k = tl.load(k_ptrs + start_n * stride_kn,
                    mask=n_col_mask[None, :], other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = tl.where(n_col_mask[None, :], qk, float("-inf"))
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        m_curr_safe = tl.where(m_curr == float("-inf"), 0.0, m_curr)
        alpha = tl.exp(m_prev - m_curr_safe)
        l_prev = l_prev * alpha
        p = tl.exp(qk - m_curr_safe[:, None])
        l_curr = tl.sum(p, 1) + l_prev

        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs + start_n * stride_vn,
                    mask=n_col_mask[:, None], other=0.0)
        p_fp16 = p.to(v.dtype)
        pv = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        pv += tl.dot(p_fp16, v)
        acc = acc + pv

        m_prev = m_curr_safe
        l_prev = l_curr

    safe_l = tl.where(l_prev > 0, l_prev, 1.0)
    acc = acc / safe_l[:, None]

    tl.store(out_ptrs, acc.to(Out.dtype.element_ty),
             mask=q_row_mask[:, None])

    if WRITE_LSE:
        # lse = m + log(l). For fully-masked rows (l==0) store +inf so that
        # exp(s - lse) → 0 in the backward pass.
        lse_val = tl.where(l_prev > 0,
                           m_prev + tl.log(safe_l),
                           float("inf"))
        lse_ptrs = Lse + off_z * stride_lz + off_h * stride_lh + offs_m
        tl.store(lse_ptrs, lse_val, mask=q_row_mask)


# -----------------------------------------------------------------------------
# Triton kernel (D = 128, two 64-halves)
# -----------------------------------------------------------------------------

@triton.autotune(configs=_FWD_CONFIGS_D128, key=['N_CTX'])
@triton.jit
def _fa_fwd_kernel_d128(
    Q, K, V, Out, Lse,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_lz, stride_lh,
    Z, H, N_CTX,
    IS_CAUSAL: tl.constexpr,
    WRITE_LSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Same as _fa_fwd_kernel_d64 but the head_dim is split into two BLOCK_D halves."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d_lo = tl.arange(0, BLOCK_D)
    offs_d_hi = BLOCK_D + tl.arange(0, BLOCK_D)

    # Pointers into Q halves
    q_ptrs_lo = Q + off_z * stride_qz + off_h * stride_qh \
        + offs_m[:, None] * stride_qm + offs_d_lo[None, :] * stride_qk
    q_ptrs_hi = Q + off_z * stride_qz + off_h * stride_qh \
        + offs_m[:, None] * stride_qm + offs_d_hi[None, :] * stride_qk
    # K halves (transposed view via strides)
    k_ptrs_lo = K + off_z * stride_kz + off_h * stride_kh \
        + offs_d_lo[:, None] * stride_kk + offs_n[None, :] * stride_kn
    k_ptrs_hi = K + off_z * stride_kz + off_h * stride_kh \
        + offs_d_hi[:, None] * stride_kk + offs_n[None, :] * stride_kn
    # V halves
    v_ptrs_lo = V + off_z * stride_vz + off_h * stride_vh \
        + offs_n[:, None] * stride_vn + offs_d_lo[None, :] * stride_vk
    v_ptrs_hi = V + off_z * stride_vz + off_h * stride_vh \
        + offs_n[:, None] * stride_vn + offs_d_hi[None, :] * stride_vk
    # Output halves
    out_ptrs_lo = Out + off_z * stride_oz + off_h * stride_oh \
        + offs_m[:, None] * stride_om + offs_d_lo[None, :] * stride_on
    out_ptrs_hi = Out + off_z * stride_oz + off_h * stride_oh \
        + offs_m[:, None] * stride_om + offs_d_hi[None, :] * stride_on

    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_lo = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    acc_hi = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    q_row_mask = offs_m < N_CTX
    q_lo = tl.load(q_ptrs_lo, mask=q_row_mask[:, None], other=0.0)
    q_hi = tl.load(q_ptrs_hi, mask=q_row_mask[:, None], other=0.0)
    # Pre-scale both Q halves once.
    q_lo = (q_lo.to(tl.float32) * sm_scale).to(q_lo.dtype)
    q_hi = (q_hi.to(tl.float32) * sm_scale).to(q_hi.dtype)

    if IS_CAUSAL:
        hi = (start_m + 1) * BLOCK_M
        if hi > N_CTX:
            hi = N_CTX
    else:
        hi = N_CTX

    for start_n in range(0, hi, BLOCK_N):
        n_col_mask = (start_n + offs_n) < N_CTX
        k_lo = tl.load(k_ptrs_lo + start_n * stride_kn,
                       mask=n_col_mask[None, :], other=0.0)
        k_hi = tl.load(k_ptrs_hi + start_n * stride_kn,
                       mask=n_col_mask[None, :], other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q_lo, k_lo)
        qk += tl.dot(q_hi, k_hi)
        qk = tl.where(n_col_mask[None, :], qk, float("-inf"))
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        m_curr_safe = tl.where(m_curr == float("-inf"), 0.0, m_curr)
        alpha = tl.exp(m_prev - m_curr_safe)
        l_prev = l_prev * alpha
        p = tl.exp(qk - m_curr_safe[:, None])
        l_curr = tl.sum(p, 1) + l_prev

        acc_lo = acc_lo * alpha[:, None]
        acc_hi = acc_hi * alpha[:, None]
        v_lo = tl.load(v_ptrs_lo + start_n * stride_vn,
                       mask=n_col_mask[:, None], other=0.0)
        v_hi = tl.load(v_ptrs_hi + start_n * stride_vn,
                       mask=n_col_mask[:, None], other=0.0)
        p_fp16 = p.to(v_lo.dtype)
        pv_lo = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        pv_hi = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        pv_lo += tl.dot(p_fp16, v_lo)
        pv_hi += tl.dot(p_fp16, v_hi)
        acc_lo = acc_lo + pv_lo
        acc_hi = acc_hi + pv_hi

        m_prev = m_curr_safe
        l_prev = l_curr

    safe_l = tl.where(l_prev > 0, l_prev, 1.0)
    acc_lo = acc_lo / safe_l[:, None]
    acc_hi = acc_hi / safe_l[:, None]

    tl.store(out_ptrs_lo, acc_lo.to(Out.dtype.element_ty), mask=q_row_mask[:, None])
    tl.store(out_ptrs_hi, acc_hi.to(Out.dtype.element_ty), mask=q_row_mask[:, None])

    if WRITE_LSE:
        lse_val = tl.where(l_prev > 0,
                           m_prev + tl.log(safe_l),
                           float("inf"))
        lse_ptrs = Lse + off_z * stride_lz + off_h * stride_lh + offs_m
        tl.store(lse_ptrs, lse_val, mask=q_row_mask)


# -----------------------------------------------------------------------------
# Python wrapper
# -----------------------------------------------------------------------------

def flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    return_softmax_lse: bool = False,
):
    """FlashAttention forward -- fp16 in/out, fp32 accumulation, V100-friendly.

    Args:
        q, k, v: (B, N, H, D) fp16 on CUDA. D in {16, 32, 64, 128}.
        causal:  if True apply lower-triangular mask.
        sm_scale: softmax scale (default 1/sqrt(D)).
        return_softmax_lse: if True, also return per-row log-sum-exp (used by
            the backward pass to recompute softmax without materialising P).

    Returns:
        out: (B, N, H, D) fp16. If return_softmax_lse, also returns
        lse: (B, H, N) fp32. For fully-masked rows lse = +inf so the backward
        recomputed exp(s - lse) is exactly zero.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "tensors must be on CUDA"
    assert q.dtype == torch.float16 == k.dtype == v.dtype, "fp16 only"
    assert q.shape == k.shape == v.shape, f"qkv shape mismatch {q.shape} {k.shape} {v.shape}"
    B, N, H, D = q.shape
    assert D in (16, 32, 64, 128), f"unsupported head_dim {D} (allowed: 16,32,64,128)"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # ---- D < 64 path: pad head_dim to 64 in-wrapper (zero-fill on outer half) ----
    if D < 64:
        D_INT = 64
        q_pad = torch.zeros((B, N, H, D_INT), dtype=q.dtype, device=q.device)
        k_pad = torch.zeros((B, N, H, D_INT), dtype=k.dtype, device=k.device)
        v_pad = torch.zeros((B, N, H, D_INT), dtype=v.dtype, device=v.device)
        q_pad[..., :D] = q
        k_pad[..., :D] = k
        v_pad[..., :D] = v
        out_full, lse = _run_kernel(q_pad, k_pad, v_pad, causal, sm_scale, D_INT,
                                    write_lse=return_softmax_lse)
        out = out_full[..., :D].contiguous()
        if return_softmax_lse:
            return out, lse
        return out

    out, lse = _run_kernel(q, k, v, causal, sm_scale, D,
                           write_lse=return_softmax_lse)
    if return_softmax_lse:
        return out, lse
    return out


def _run_kernel(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    causal: bool, sm_scale: float, D_INT: int,
    write_lse: bool = False,
):
    """Permute, dispatch to the correct kernel, permute back.

    Returns (out, lse) where lse is None unless write_lse=True. lse layout is
    (B, H, N) fp32 — kept in (B, H, N) order to match the kernel grid (one
    program writes a contiguous BLOCK_M-row stripe per (batch, head)).
    """
    B, N, H, _ = q.shape
    # (B,N,H,D) -> (B,H,N,D); contiguous for predictable strides.
    q_ = q.permute(0, 2, 1, 3).contiguous()
    k_ = k.permute(0, 2, 1, 3).contiguous()
    v_ = v.permute(0, 2, 1, 3).contiguous()
    out_ = torch.empty_like(q_)

    if write_lse:
        lse = torch.empty((B, H, N), dtype=torch.float32, device=q.device)
        lse_stride_z, lse_stride_h = lse.stride(0), lse.stride(1)
    else:
        # Triton needs a real pointer even when WRITE_LSE=False; allocate the
        # smallest legal tensor and pass zero strides (kernel never reads it).
        lse = torch.empty((1,), dtype=torch.float32, device=q.device)
        lse_stride_z, lse_stride_h = 0, 0

    if D_INT == 64:
        kernel = _fa_fwd_kernel_d64
    elif D_INT == 128:
        kernel = _fa_fwd_kernel_d128
    else:
        raise AssertionError(f"unreachable D_INT={D_INT}")

    # Grid depends on the autotuned BLOCK_M, so use a callable form.
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_M']), B * H, 1)
    kernel[grid](
        q_, k_, v_, out_, lse,
        sm_scale,
        q_.stride(0), q_.stride(1), q_.stride(2), q_.stride(3),
        k_.stride(0), k_.stride(1), k_.stride(2), k_.stride(3),
        v_.stride(0), v_.stride(1), v_.stride(2), v_.stride(3),
        out_.stride(0), out_.stride(1), out_.stride(2), out_.stride(3),
        lse_stride_z, lse_stride_h,
        B, H, N,
        IS_CAUSAL=causal,
        WRITE_LSE=write_lse,
    )

    out = out_.permute(0, 2, 1, 3).contiguous()
    if write_lse:
        return out, lse  # lse is (B, H, N) fp32
    return out, None


# Public alias
flash_attn_volta = flash_attn_forward
