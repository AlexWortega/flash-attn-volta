"""Triton FlashAttention backward kernels for NVIDIA Volta (SM 7.0 / V100).

Implements FlashAttention-1 backward (Dao et al., 2205.14135 §B / Algo 4) as
two separate Triton 2.3 kernels — one for dQ (parallel over Q-blocks) and one
for dK/dV (parallel over K-blocks). Both kernels recompute S and P from the
saved LSE; no O(N²) intermediate is materialised.

External API:
    flash_attn_backward(do, q, k, v, o, lse, causal=False, sm_scale=None)
        -> (dq, dk, dv)

Shapes:
    do, q, k, v, o : (B, N, H, D)   fp16  (D in {16, 32, 64, 128})
    lse            : (B, H, N)      fp32   (from flash_attn_forward(..., return_softmax_lse=True))
    dq, dk, dv     : (B, N, H, D)   fp16

Volta / Triton-2.3 quirk: every `tl.dot` K-dim is kept at 64 (BLOCK_M = BLOCK_N
= BLOCK_D = 64). For D = 128 we split the head into two 64-halves and run two
`tl.dot`s exactly like the forward. D < 64 is padded to 64 in the wrapper.

Numerical stability: for fully-masked rows the forward stores LSE = +inf so
that exp(s - LSE) = 0 here without any extra branch.
"""

from __future__ import annotations
import math

import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Autotune config sets (mirror forward constraints: BN=64, nw=4, BLOCK_D=64).
# -----------------------------------------------------------------------------
#
# Backward keeps Q + dO (and dK/dV also Q^T + dO^T) resident per iteration, so
# SMEM is tighter than forward. We sweep BLOCK_M and num_stages only — same
# Triton-2.3 V100 quirks apply (BN ∈ {32, 128} miscompiles, num_warps=8 emits
# wrong output, BM is the K-dim of Q^T @ dS so must be 64).
#
# Note: BM = 64 (not 128) on backward because BM is a K-dim in two of the
# backward dots; widening it would force re-loading Q^T at twice the SMEM
# cost, and BM=128 spilled badly in probes.

_BWD_DQ_CONFIGS_D64 = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=1),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=3),
]

_BWD_DQ_CONFIGS_D128 = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=1),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
]

_BWD_DKDV_CONFIGS_D64 = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=1),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=3),
]

_BWD_DKDV_CONFIGS_D128 = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=1),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=4, num_stages=2),
]


# -----------------------------------------------------------------------------
# dQ kernel — D = 64
# -----------------------------------------------------------------------------

@triton.autotune(configs=_BWD_DQ_CONFIGS_D64, key=['N_CTX'])
@triton.jit
def _fa_bwd_dq_kernel_d64(
    Q, K, V, DO, DQ, LSE, D_VEC,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dod,
    stride_dqz, stride_dqh, stride_dqm, stride_dqd,
    stride_lz, stride_lh,
    stride_dz, stride_dh,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """One program = one (Q-block i, batch_head). Streams K/V tiles, accumulates dQ."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_row_mask = offs_m < N_CTX

    # Pointers
    q_ptrs = Q + off_z * stride_qz + off_h * stride_qh \
        + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    do_ptrs = DO + off_z * stride_doz + off_h * stride_doh \
        + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
    dq_ptrs = DQ + off_z * stride_dqz + off_h * stride_dqh \
        + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
    # K^T-shaped pointers for q @ k^T → use d as outer, n as inner
    k_ptrs = K + off_z * stride_kz + off_h * stride_kh \
        + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    # V^T-shaped pointers for do @ v^T
    v_ptrs = V + off_z * stride_vz + off_h * stride_vh \
        + offs_d[:, None] * stride_vk + offs_n[None, :] * stride_vn
    # K-shaped pointers for ds @ k → use n as outer, d as inner
    k_for_dq_ptrs = K + off_z * stride_kz + off_h * stride_kh \
        + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk

    # Load Q, dO, LSE, D — once.
    q = tl.load(q_ptrs, mask=q_row_mask[:, None], other=0.0)
    # Pre-scale Q by sm_scale so q @ k^T already equals S = sm_scale * Q @ K^T.
    q_scaled = (q.to(tl.float32) * sm_scale).to(q.dtype)
    do = tl.load(do_ptrs, mask=q_row_mask[:, None], other=0.0)

    lse_ptrs = LSE + off_z * stride_lz + off_h * stride_lh + offs_m
    # other = +inf so masked rows get p = exp(s - inf) = 0
    lse_i = tl.load(lse_ptrs, mask=q_row_mask, other=float("inf"))
    d_ptrs = D_VEC + off_z * stride_dz + off_h * stride_dh + offs_m
    d_i = tl.load(d_ptrs, mask=q_row_mask, other=0.0)

    # dQ accumulator (fp32). Will be scaled by sm_scale at the end.
    dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    if IS_CAUSAL:
        hi = (start_m + 1) * BLOCK_M
        if hi > N_CTX:
            hi = N_CTX
    else:
        hi = N_CTX

    for start_n in range(0, hi, BLOCK_N):
        n_col_mask = (start_n + offs_n) < N_CTX

        # K block (transposed view via offs_d[:,None]·stride_kk)
        k = tl.load(k_ptrs + start_n * stride_kn,
                    mask=n_col_mask[None, :], other=0.0)
        # V block (transposed)
        v = tl.load(v_ptrs + start_n * stride_vn,
                    mask=n_col_mask[None, :], other=0.0)
        # K block in (n, d) layout for ds @ k
        k_for_dq = tl.load(k_for_dq_ptrs + start_n * stride_kn,
                           mask=n_col_mask[:, None], other=0.0)

        # s = q_scaled @ k^T  (fp32 accumulator)
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q_scaled, k)
        # mask invalid columns
        s = tl.where(n_col_mask[None, :], s, float("-inf"))
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            s = tl.where(causal_mask, s, float("-inf"))

        # p = exp(s - lse[i])
        p = tl.exp(s - lse_i[:, None])

        # dp = do @ v^T  (fp32)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, v)

        # ds = p * (dp - D[i])
        ds = p * (dp - d_i[:, None])

        # dQ += ds @ K   (K is in (n, d) layout)
        ds_fp16 = ds.to(k_for_dq.dtype)
        dq_acc += tl.dot(ds_fp16, k_for_dq)

    dq_acc = dq_acc * sm_scale
    tl.store(dq_ptrs, dq_acc.to(DQ.dtype.element_ty),
             mask=q_row_mask[:, None])


# -----------------------------------------------------------------------------
# dQ kernel — D = 128 (split into two 64-halves)
# -----------------------------------------------------------------------------

@triton.autotune(configs=_BWD_DQ_CONFIGS_D128, key=['N_CTX'])
@triton.jit
def _fa_bwd_dq_kernel_d128(
    Q, K, V, DO, DQ, LSE, D_VEC,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dod,
    stride_dqz, stride_dqh, stride_dqm, stride_dqd,
    stride_lz, stride_lh,
    stride_dz, stride_dh,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d_lo = tl.arange(0, BLOCK_D)
    offs_d_hi = BLOCK_D + tl.arange(0, BLOCK_D)

    q_row_mask = offs_m < N_CTX

    # Q halves
    q_ptrs_lo = Q + off_z * stride_qz + off_h * stride_qh \
        + offs_m[:, None] * stride_qm + offs_d_lo[None, :] * stride_qk
    q_ptrs_hi = Q + off_z * stride_qz + off_h * stride_qh \
        + offs_m[:, None] * stride_qm + offs_d_hi[None, :] * stride_qk
    do_ptrs_lo = DO + off_z * stride_doz + off_h * stride_doh \
        + offs_m[:, None] * stride_dom + offs_d_lo[None, :] * stride_dod
    do_ptrs_hi = DO + off_z * stride_doz + off_h * stride_doh \
        + offs_m[:, None] * stride_dom + offs_d_hi[None, :] * stride_dod
    dq_ptrs_lo = DQ + off_z * stride_dqz + off_h * stride_dqh \
        + offs_m[:, None] * stride_dqm + offs_d_lo[None, :] * stride_dqd
    dq_ptrs_hi = DQ + off_z * stride_dqz + off_h * stride_dqh \
        + offs_m[:, None] * stride_dqm + offs_d_hi[None, :] * stride_dqd

    # K^T-shaped pointers (d outer, n inner) for q @ k^T
    k_ptrs_lo = K + off_z * stride_kz + off_h * stride_kh \
        + offs_d_lo[:, None] * stride_kk + offs_n[None, :] * stride_kn
    k_ptrs_hi = K + off_z * stride_kz + off_h * stride_kh \
        + offs_d_hi[:, None] * stride_kk + offs_n[None, :] * stride_kn
    # V^T-shaped pointers (d outer, n inner) for do @ v^T
    v_ptrs_lo = V + off_z * stride_vz + off_h * stride_vh \
        + offs_d_lo[:, None] * stride_vk + offs_n[None, :] * stride_vn
    v_ptrs_hi = V + off_z * stride_vz + off_h * stride_vh \
        + offs_d_hi[:, None] * stride_vk + offs_n[None, :] * stride_vn
    # K (n, d) layout for ds @ k
    k_for_dq_ptrs_lo = K + off_z * stride_kz + off_h * stride_kh \
        + offs_n[:, None] * stride_kn + offs_d_lo[None, :] * stride_kk
    k_for_dq_ptrs_hi = K + off_z * stride_kz + off_h * stride_kh \
        + offs_n[:, None] * stride_kn + offs_d_hi[None, :] * stride_kk

    q_lo = tl.load(q_ptrs_lo, mask=q_row_mask[:, None], other=0.0)
    q_hi = tl.load(q_ptrs_hi, mask=q_row_mask[:, None], other=0.0)
    q_lo_scaled = (q_lo.to(tl.float32) * sm_scale).to(q_lo.dtype)
    q_hi_scaled = (q_hi.to(tl.float32) * sm_scale).to(q_hi.dtype)
    do_lo = tl.load(do_ptrs_lo, mask=q_row_mask[:, None], other=0.0)
    do_hi = tl.load(do_ptrs_hi, mask=q_row_mask[:, None], other=0.0)

    lse_ptrs = LSE + off_z * stride_lz + off_h * stride_lh + offs_m
    lse_i = tl.load(lse_ptrs, mask=q_row_mask, other=float("inf"))
    d_ptrs = D_VEC + off_z * stride_dz + off_h * stride_dh + offs_m
    d_i = tl.load(d_ptrs, mask=q_row_mask, other=0.0)

    dq_acc_lo = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    dq_acc_hi = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

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
        v_lo = tl.load(v_ptrs_lo + start_n * stride_vn,
                       mask=n_col_mask[None, :], other=0.0)
        v_hi = tl.load(v_ptrs_hi + start_n * stride_vn,
                       mask=n_col_mask[None, :], other=0.0)
        k_for_dq_lo = tl.load(k_for_dq_ptrs_lo + start_n * stride_kn,
                              mask=n_col_mask[:, None], other=0.0)
        k_for_dq_hi = tl.load(k_for_dq_ptrs_hi + start_n * stride_kn,
                              mask=n_col_mask[:, None], other=0.0)

        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q_lo_scaled, k_lo)
        s += tl.dot(q_hi_scaled, k_hi)
        s = tl.where(n_col_mask[None, :], s, float("-inf"))
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            s = tl.where(causal_mask, s, float("-inf"))

        p = tl.exp(s - lse_i[:, None])

        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do_lo, v_lo)
        dp += tl.dot(do_hi, v_hi)

        ds = p * (dp - d_i[:, None])
        ds_fp16 = ds.to(k_for_dq_lo.dtype)
        dq_acc_lo += tl.dot(ds_fp16, k_for_dq_lo)
        dq_acc_hi += tl.dot(ds_fp16, k_for_dq_hi)

    dq_acc_lo = dq_acc_lo * sm_scale
    dq_acc_hi = dq_acc_hi * sm_scale
    tl.store(dq_ptrs_lo, dq_acc_lo.to(DQ.dtype.element_ty), mask=q_row_mask[:, None])
    tl.store(dq_ptrs_hi, dq_acc_hi.to(DQ.dtype.element_ty), mask=q_row_mask[:, None])


# -----------------------------------------------------------------------------
# dK/dV kernel — D = 64
#
# To avoid `tl.trans` (which the Triton-2.3 V100 backend mis-compiles), we
# accumulate dK^T and dV^T in (BD, BN) layout and store them transposed via
# strided pointers. The required q^T / dO^T blocks are loaded from HBM with
# swapped strides — same data, no in-kernel transpose.
# -----------------------------------------------------------------------------

@triton.autotune(configs=_BWD_DKDV_CONFIGS_D64, key=['N_CTX'])
@triton.jit
def _fa_bwd_dkdv_kernel_d64(
    Q, K, V, DO, DK, DV, LSE, D_VEC,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dod,
    stride_dkz, stride_dkh, stride_dkn, stride_dkd,
    stride_dvz, stride_dvh, stride_dvn, stride_dvd,
    stride_lz, stride_lh,
    stride_dz, stride_dh,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """One program = one (K-block j, batch_head). Streams Q/dO tiles, accumulates dK and dV."""
    start_n = tl.program_id(0)  # K-block index along seq
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    n_row_mask = offs_n < N_CTX

    # K^T (d, n) layout for q @ k^T inside the loop
    k_t_ptrs = K + off_z * stride_kz + off_h * stride_kh \
        + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    k_t = tl.load(k_t_ptrs, mask=n_row_mask[None, :], other=0.0)
    # Pre-scale K^T (used inside q @ k^T) once: makes s = q @ k_t_scaled equal S.
    k_t_scaled = (k_t.to(tl.float32) * sm_scale).to(k_t.dtype)

    # V^T (d, n) for dp = do @ v^T
    v_t_ptrs = V + off_z * stride_vz + off_h * stride_vh \
        + offs_d[:, None] * stride_vk + offs_n[None, :] * stride_vn
    v_t = tl.load(v_t_ptrs, mask=n_row_mask[None, :], other=0.0)

    q_base = Q + off_z * stride_qz + off_h * stride_qh
    do_base = DO + off_z * stride_doz + off_h * stride_doh
    lse_base = LSE + off_z * stride_lz + off_h * stride_lh
    d_base = D_VEC + off_z * stride_dz + off_h * stride_dh

    # We accumulate transposed: dK^T[d, n] and dV^T[d, n]. Final store
    # transposes via strided pointers.
    dk_t_acc = tl.zeros([BLOCK_D, BLOCK_N], dtype=tl.float32)
    dv_t_acc = tl.zeros([BLOCK_D, BLOCK_N], dtype=tl.float32)

    if IS_CAUSAL:
        # First Q-block whose row-range overlaps any unmasked column in this K-block.
        # With BLOCK_M == BLOCK_N this is start_n; we keep the explicit form
        # so it generalises if the sizes ever diverge.
        lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M
    else:
        lo = 0

    for start_m in range(lo, N_CTX, BLOCK_M):
        m_row_mask = (start_m + offs_m) < N_CTX

        # Q (m, d) for s = q @ k^T
        q_ptrs = q_base + (start_m + offs_m)[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q = tl.load(q_ptrs, mask=m_row_mask[:, None], other=0.0)
        # Q^T (d, m) for dK^T += q^T @ ds
        q_t_ptrs = q_base + offs_d[:, None] * stride_qk + (start_m + offs_m)[None, :] * stride_qm
        q_t = tl.load(q_t_ptrs, mask=m_row_mask[None, :], other=0.0)
        # dO (m, d) for dp = do @ v^T
        do_ptrs = do_base + (start_m + offs_m)[:, None] * stride_dom + offs_d[None, :] * stride_dod
        do = tl.load(do_ptrs, mask=m_row_mask[:, None], other=0.0)
        # dO^T (d, m) for dV^T += do^T @ p
        do_t_ptrs = do_base + offs_d[:, None] * stride_dod + (start_m + offs_m)[None, :] * stride_dom
        do_t = tl.load(do_t_ptrs, mask=m_row_mask[None, :], other=0.0)

        lse_i = tl.load(lse_base + (start_m + offs_m),
                        mask=m_row_mask, other=float("inf"))
        d_i = tl.load(d_base + (start_m + offs_m),
                      mask=m_row_mask, other=0.0)

        # s = q @ (sm_scale * k^T)  — sm_scale baked into k_t_scaled
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, k_t_scaled)
        s = tl.where(n_row_mask[None, :], s, float("-inf"))
        s = tl.where(m_row_mask[:, None], s, float("-inf"))
        if IS_CAUSAL:
            row_idx = (start_m + offs_m)[:, None]
            col_idx = offs_n[None, :]
            s = tl.where(row_idx >= col_idx, s, float("-inf"))

        p = tl.exp(s - lse_i[:, None])  # (BM, BN), fp32
        p_fp16 = p.to(do_t.dtype)

        # dV^T += dO^T @ P    (BD, BM) @ (BM, BN) → (BD, BN)
        dv_t_acc += tl.dot(do_t, p_fp16)

        # dp = dO @ V^T   (BM, BD) @ (BD, BN) → (BM, BN)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, v_t)

        # ds = p * (dp - D[i])
        ds = p * (dp - d_i[:, None])
        ds_fp16 = ds.to(q_t.dtype)

        # dK^T += Q^T @ dS    (BD, BM) @ (BM, BN) → (BD, BN)
        dk_t_acc += tl.dot(q_t, ds_fp16)

    # Apply sm_scale to dK only (dV is unscaled — V is not in S).
    dk_t_acc = dk_t_acc * sm_scale

    # Store dK^T (BD, BN) transposed into DK (BN, BD): swap the index roles.
    dk_ptrs = DK + off_z * stride_dkz + off_h * stride_dkh \
        + offs_n[None, :] * stride_dkn + offs_d[:, None] * stride_dkd
    dv_ptrs = DV + off_z * stride_dvz + off_h * stride_dvh \
        + offs_n[None, :] * stride_dvn + offs_d[:, None] * stride_dvd
    tl.store(dk_ptrs, dk_t_acc.to(DK.dtype.element_ty), mask=n_row_mask[None, :])
    tl.store(dv_ptrs, dv_t_acc.to(DV.dtype.element_ty), mask=n_row_mask[None, :])


# -----------------------------------------------------------------------------
# dK/dV kernel — D = 128 (split into two 64-halves)
# -----------------------------------------------------------------------------

@triton.autotune(configs=_BWD_DKDV_CONFIGS_D128, key=['N_CTX'])
@triton.jit
def _fa_bwd_dkdv_kernel_d128(
    Q, K, V, DO, DK, DV, LSE, D_VEC,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dod,
    stride_dkz, stride_dkh, stride_dkn, stride_dkd,
    stride_dvz, stride_dvh, stride_dvn, stride_dvd,
    stride_lz, stride_lh,
    stride_dz, stride_dh,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d_lo = tl.arange(0, BLOCK_D)
    offs_d_hi = BLOCK_D + tl.arange(0, BLOCK_D)

    n_row_mask = offs_n < N_CTX

    # K^T (d, n) for q @ k^T
    k_t_ptrs_lo = K + off_z * stride_kz + off_h * stride_kh \
        + offs_d_lo[:, None] * stride_kk + offs_n[None, :] * stride_kn
    k_t_ptrs_hi = K + off_z * stride_kz + off_h * stride_kh \
        + offs_d_hi[:, None] * stride_kk + offs_n[None, :] * stride_kn
    k_t_lo = tl.load(k_t_ptrs_lo, mask=n_row_mask[None, :], other=0.0)
    k_t_hi = tl.load(k_t_ptrs_hi, mask=n_row_mask[None, :], other=0.0)
    k_t_lo_scaled = (k_t_lo.to(tl.float32) * sm_scale).to(k_t_lo.dtype)
    k_t_hi_scaled = (k_t_hi.to(tl.float32) * sm_scale).to(k_t_hi.dtype)

    # V^T (d, n) for dp = do @ v^T
    v_t_ptrs_lo = V + off_z * stride_vz + off_h * stride_vh \
        + offs_d_lo[:, None] * stride_vk + offs_n[None, :] * stride_vn
    v_t_ptrs_hi = V + off_z * stride_vz + off_h * stride_vh \
        + offs_d_hi[:, None] * stride_vk + offs_n[None, :] * stride_vn
    v_t_lo = tl.load(v_t_ptrs_lo, mask=n_row_mask[None, :], other=0.0)
    v_t_hi = tl.load(v_t_ptrs_hi, mask=n_row_mask[None, :], other=0.0)

    q_base = Q + off_z * stride_qz + off_h * stride_qh
    do_base = DO + off_z * stride_doz + off_h * stride_doh
    lse_base = LSE + off_z * stride_lz + off_h * stride_lh
    d_base = D_VEC + off_z * stride_dz + off_h * stride_dh

    # Accumulate transposed (BD, BN) — stored transposed to DK/DV at end.
    dk_t_acc_lo = tl.zeros([BLOCK_D, BLOCK_N], dtype=tl.float32)
    dk_t_acc_hi = tl.zeros([BLOCK_D, BLOCK_N], dtype=tl.float32)
    dv_t_acc_lo = tl.zeros([BLOCK_D, BLOCK_N], dtype=tl.float32)
    dv_t_acc_hi = tl.zeros([BLOCK_D, BLOCK_N], dtype=tl.float32)

    if IS_CAUSAL:
        lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M
    else:
        lo = 0

    for start_m in range(lo, N_CTX, BLOCK_M):
        m_row_mask = (start_m + offs_m) < N_CTX

        # ---- Phase 1: load Q/dO (m,d) → compute s, p, dp, ds. ----
        q_ptrs_lo = q_base + (start_m + offs_m)[:, None] * stride_qm + offs_d_lo[None, :] * stride_qk
        q_ptrs_hi = q_base + (start_m + offs_m)[:, None] * stride_qm + offs_d_hi[None, :] * stride_qk
        q_lo = tl.load(q_ptrs_lo, mask=m_row_mask[:, None], other=0.0)
        q_hi = tl.load(q_ptrs_hi, mask=m_row_mask[:, None], other=0.0)
        do_ptrs_lo = do_base + (start_m + offs_m)[:, None] * stride_dom + offs_d_lo[None, :] * stride_dod
        do_ptrs_hi = do_base + (start_m + offs_m)[:, None] * stride_dom + offs_d_hi[None, :] * stride_dod
        do_lo = tl.load(do_ptrs_lo, mask=m_row_mask[:, None], other=0.0)
        do_hi = tl.load(do_ptrs_hi, mask=m_row_mask[:, None], other=0.0)

        lse_i = tl.load(lse_base + (start_m + offs_m),
                        mask=m_row_mask, other=float("inf"))
        d_i = tl.load(d_base + (start_m + offs_m),
                      mask=m_row_mask, other=0.0)

        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q_lo, k_t_lo_scaled)
        s += tl.dot(q_hi, k_t_hi_scaled)
        s = tl.where(n_row_mask[None, :], s, float("-inf"))
        s = tl.where(m_row_mask[:, None], s, float("-inf"))
        if IS_CAUSAL:
            row_idx = (start_m + offs_m)[:, None]
            col_idx = offs_n[None, :]
            s = tl.where(row_idx >= col_idx, s, float("-inf"))

        p = tl.exp(s - lse_i[:, None])

        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do_lo, v_t_lo)
        dp += tl.dot(do_hi, v_t_hi)

        ds = p * (dp - d_i[:, None])

        # q_lo/q_hi/do_lo/do_hi are dead from here on. Cast p, ds to fp16 now
        # and free the originals before loading the transposes.
        p_fp16 = p.to(q_lo.dtype)
        ds_fp16 = ds.to(q_lo.dtype)

        # ---- Phase 2: load Q^T / dO^T (d,m) → accumulate dV^T, dK^T. ----
        do_t_ptrs_lo = do_base + offs_d_lo[:, None] * stride_dod + (start_m + offs_m)[None, :] * stride_dom
        do_t_ptrs_hi = do_base + offs_d_hi[:, None] * stride_dod + (start_m + offs_m)[None, :] * stride_dom
        do_t_lo = tl.load(do_t_ptrs_lo, mask=m_row_mask[None, :], other=0.0)
        do_t_hi = tl.load(do_t_ptrs_hi, mask=m_row_mask[None, :], other=0.0)
        dv_t_acc_lo += tl.dot(do_t_lo, p_fp16)
        dv_t_acc_hi += tl.dot(do_t_hi, p_fp16)

        q_t_ptrs_lo = q_base + offs_d_lo[:, None] * stride_qk + (start_m + offs_m)[None, :] * stride_qm
        q_t_ptrs_hi = q_base + offs_d_hi[:, None] * stride_qk + (start_m + offs_m)[None, :] * stride_qm
        q_t_lo = tl.load(q_t_ptrs_lo, mask=m_row_mask[None, :], other=0.0)
        q_t_hi = tl.load(q_t_ptrs_hi, mask=m_row_mask[None, :], other=0.0)
        dk_t_acc_lo += tl.dot(q_t_lo, ds_fp16)
        dk_t_acc_hi += tl.dot(q_t_hi, ds_fp16)

    dk_t_acc_lo = dk_t_acc_lo * sm_scale
    dk_t_acc_hi = dk_t_acc_hi * sm_scale

    # Transposed stores: (BD, BN) → DK[BN, BD]
    dk_ptrs_lo = DK + off_z * stride_dkz + off_h * stride_dkh \
        + offs_n[None, :] * stride_dkn + offs_d_lo[:, None] * stride_dkd
    dk_ptrs_hi = DK + off_z * stride_dkz + off_h * stride_dkh \
        + offs_n[None, :] * stride_dkn + offs_d_hi[:, None] * stride_dkd
    dv_ptrs_lo = DV + off_z * stride_dvz + off_h * stride_dvh \
        + offs_n[None, :] * stride_dvn + offs_d_lo[:, None] * stride_dvd
    dv_ptrs_hi = DV + off_z * stride_dvz + off_h * stride_dvh \
        + offs_n[None, :] * stride_dvn + offs_d_hi[:, None] * stride_dvd
    tl.store(dk_ptrs_lo, dk_t_acc_lo.to(DK.dtype.element_ty), mask=n_row_mask[None, :])
    tl.store(dk_ptrs_hi, dk_t_acc_hi.to(DK.dtype.element_ty), mask=n_row_mask[None, :])
    tl.store(dv_ptrs_lo, dv_t_acc_lo.to(DV.dtype.element_ty), mask=n_row_mask[None, :])
    tl.store(dv_ptrs_hi, dv_t_acc_hi.to(DV.dtype.element_ty), mask=n_row_mask[None, :])


# -----------------------------------------------------------------------------
# Python wrapper
# -----------------------------------------------------------------------------

def flash_attn_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashAttention backward — fp16 in/out, fp32 accumulation.

    Args:
        do: (B, N, H, D) fp16, gradient w.r.t. output.
        q, k, v, o: (B, N, H, D) fp16. q,k,v are forward inputs; o is forward output.
        lse: (B, H, N) fp32, log-sum-exp from forward (with +inf in fully-masked rows).
        causal: must match the forward causal flag.
        sm_scale: must match the forward sm_scale.

    Returns:
        (dq, dk, dv) — each (B, N, H, D) fp16.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda and o.is_cuda and do.is_cuda
    assert q.dtype == torch.float16 == k.dtype == v.dtype == o.dtype == do.dtype
    assert lse.dtype == torch.float32
    assert q.shape == k.shape == v.shape == o.shape == do.shape
    B, N, H, D = q.shape
    assert D in (16, 32, 64, 128), f"unsupported head_dim {D}"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # Pad D < 64 → 64 (matches forward padding).
    if D < 64:
        D_INT = 64
        def pad(t):
            tp = torch.zeros((B, N, H, D_INT), dtype=t.dtype, device=t.device)
            tp[..., :D] = t
            return tp
        q_p, k_p, v_p, o_p, do_p = pad(q), pad(k), pad(v), pad(o), pad(do)
        dq_p, dk_p, dv_p = _run_bwd_kernels(do_p, q_p, k_p, v_p, o_p, lse,
                                            causal, sm_scale, D_INT)
        return (dq_p[..., :D].contiguous(),
                dk_p[..., :D].contiguous(),
                dv_p[..., :D].contiguous())

    return _run_bwd_kernels(do, q, k, v, o, lse, causal, sm_scale, D)


def _run_bwd_kernels(do, q, k, v, o, lse, causal, sm_scale, D_INT):
    B, N, H, _ = q.shape

    # (B, N, H, D) → (B, H, N, D)
    q_  = q .permute(0, 2, 1, 3).contiguous()
    k_  = k .permute(0, 2, 1, 3).contiguous()
    v_  = v .permute(0, 2, 1, 3).contiguous()
    do_ = do.permute(0, 2, 1, 3).contiguous()
    o_  = o .permute(0, 2, 1, 3).contiguous()
    # lse comes in as (B, H, N) already (forward writes it that way).
    assert lse.shape == (B, H, N), f"lse shape {lse.shape} expected ({B},{H},{N})"
    lse_c = lse.contiguous()

    # D[i] = rowsum(O * dO).   (B, H, N) fp32.
    d_vec = (o_.float() * do_.float()).sum(-1).contiguous()

    dq_ = torch.empty_like(q_)
    dk_ = torch.empty_like(k_)
    dv_ = torch.empty_like(v_)

    if D_INT == 64:
        kfn_dq, kfn_dkdv = _fa_bwd_dq_kernel_d64, _fa_bwd_dkdv_kernel_d64
    elif D_INT == 128:
        kfn_dq, kfn_dkdv = _fa_bwd_dq_kernel_d128, _fa_bwd_dkdv_kernel_d128
    else:
        raise AssertionError(f"unreachable D_INT={D_INT}")

    grid_dq   = lambda meta: (triton.cdiv(N, meta['BLOCK_M']), B * H, 1)
    grid_dkdv = lambda meta: (triton.cdiv(N, meta['BLOCK_N']), B * H, 1)

    # dQ kernel
    kfn_dq[grid_dq](
        q_, k_, v_, do_, dq_, lse_c, d_vec,
        sm_scale,
        q_.stride(0),  q_.stride(1),  q_.stride(2),  q_.stride(3),
        k_.stride(0),  k_.stride(1),  k_.stride(2),  k_.stride(3),
        v_.stride(0),  v_.stride(1),  v_.stride(2),  v_.stride(3),
        do_.stride(0), do_.stride(1), do_.stride(2), do_.stride(3),
        dq_.stride(0), dq_.stride(1), dq_.stride(2), dq_.stride(3),
        lse_c.stride(0), lse_c.stride(1),
        d_vec.stride(0), d_vec.stride(1),
        B, H, N,
        IS_CAUSAL=causal,
    )

    # dK/dV kernel
    kfn_dkdv[grid_dkdv](
        q_, k_, v_, do_, dk_, dv_, lse_c, d_vec,
        sm_scale,
        q_.stride(0),  q_.stride(1),  q_.stride(2),  q_.stride(3),
        k_.stride(0),  k_.stride(1),  k_.stride(2),  k_.stride(3),
        v_.stride(0),  v_.stride(1),  v_.stride(2),  v_.stride(3),
        do_.stride(0), do_.stride(1), do_.stride(2), do_.stride(3),
        dk_.stride(0), dk_.stride(1), dk_.stride(2), dk_.stride(3),
        dv_.stride(0), dv_.stride(1), dv_.stride(2), dv_.stride(3),
        lse_c.stride(0), lse_c.stride(1),
        d_vec.stride(0), d_vec.stride(1),
        B, H, N,
        IS_CAUSAL=causal,
    )

    dq = dq_.permute(0, 2, 1, 3).contiguous()
    dk = dk_.permute(0, 2, 1, 3).contiguous()
    dv = dv_.permute(0, 2, 1, 3).contiguous()
    return dq, dk, dv
