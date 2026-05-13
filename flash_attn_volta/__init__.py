"""flash-attn-volta — FlashAttention forward kernel for NVIDIA Volta (SM 7.0).

Public API:
    flash_attn_forward(q, k, v, causal=False, sm_scale=None) -> out

Tensors are (batch, seq, n_heads, head_dim) fp16, output is fp16.
"""

from .triton_fa import flash_attn_forward, flash_attn_volta

__all__ = ["flash_attn_forward", "flash_attn_volta"]
__version__ = "0.1.0"
