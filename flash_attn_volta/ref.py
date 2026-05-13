"""Reference fp32 attention for correctness comparison."""

from __future__ import annotations
import math

import torch


def attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    dtype_out: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Standard (B, N, H, D) attention computed in fp32, cast back."""
    B, N, H, D = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    q_ = q.permute(0, 2, 1, 3).float()  # (B, H, N, D)
    k_ = k.permute(0, 2, 1, 3).float()
    v_ = v.permute(0, 2, 1, 3).float()

    scores = torch.einsum("bhnd,bhmd->bhnm", q_, k_) * sm_scale
    if causal:
        mask = torch.triu(
            torch.full((N, N), float("-inf"), device=q.device, dtype=torch.float32),
            diagonal=1,
        )
        scores = scores + mask
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhnm,bhmd->bhnd", probs, v_)
    out = out.permute(0, 2, 1, 3).contiguous().to(dtype_out)
    return out
