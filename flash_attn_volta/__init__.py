"""flash-attn-volta — FlashAttention forward + backward kernels for Volta (SM 7.0).

Public API:
    flash_attn_forward(q, k, v, causal=False, sm_scale=None,
                       return_softmax_lse=False)         -> out [, lse]
    flash_attn(q, k, v, causal=False, sm_scale=None)     -> out  (autograd-aware)
    flash_attn_backward(do, q, k, v, o, lse,
                        causal=False, sm_scale=None)     -> (dq, dk, dv)

Tensors are (batch, seq, n_heads, head_dim) fp16, output is fp16. LSE is
(B, H, N) fp32.
"""

from .triton_fa import flash_attn_forward, flash_attn_volta
from .triton_fa_bwd import flash_attn_backward
from .autograd import flash_attn

__all__ = [
    "flash_attn_forward",
    "flash_attn_volta",
    "flash_attn_backward",
    "flash_attn",
]
__version__ = "0.2.0"
