"""Autograd-aware FlashAttention entry point.

`flash_attn(q, k, v, causal, sm_scale)` calls the forward Triton kernel and
saves `(q, k, v, o, lse)` for the backward pass, which dispatches to the two
backward Triton kernels (dQ + dK/dV) defined in `triton_fa_bwd`.

The non-autograd raw entry-point `flash_attn_forward` is unchanged.
"""

from __future__ import annotations
import math

import torch

from .triton_fa import flash_attn_forward
from .triton_fa_bwd import flash_attn_backward


class _FlashAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal: bool, sm_scale: float | None):
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.shape[-1])
        out, lse = flash_attn_forward(q, k, v,
                                      causal=causal, sm_scale=sm_scale,
                                      return_softmax_lse=True)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, lse = ctx.saved_tensors
        # Make do contiguous fp16 (matches saved tensors). Upstream code may
        # hand us a non-contiguous view; the kernel needs predictable strides.
        do = do.contiguous()
        if do.dtype != q.dtype:
            do = do.to(q.dtype)
        dq, dk, dv = flash_attn_backward(do, q, k, v, out, lse,
                                         causal=ctx.causal,
                                         sm_scale=ctx.sm_scale)
        # forward signature: (q, k, v, causal, sm_scale) → grads of same arity
        return dq, dk, dv, None, None


def flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               causal: bool = False, sm_scale: float | None = None) -> torch.Tensor:
    """Autograd-aware FlashAttention. Drop-in for `F.scaled_dot_product_attention`.

    Args:
        q, k, v: (B, N, H, D) fp16 on CUDA. D in {16, 32, 64, 128}.
        causal:  lower-triangular mask if True.
        sm_scale: softmax scale (default 1/sqrt(D)).

    Returns:
        out: (B, N, H, D) fp16, with autograd hooked up.
    """
    return _FlashAttnFn.apply(q, k, v, causal, sm_scale)
