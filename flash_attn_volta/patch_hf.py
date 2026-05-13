"""Monkey-patches that route HuggingFace attention through ``flash_attn_volta``.

Two patches:

* :func:`patch_gpt2(model)` -- replaces ``GPT2Attention._attn`` on every layer.
  GPT-2 is multi-head (no GQA); inputs to ``_attn`` are ``(B, H, N, D)``.

* :func:`patch_qwen2(model)` -- replaces the ``forward`` of every Qwen2 self-
  attention module (the ``sdpa`` / ``eager`` variant present on the loaded
  model). Qwen2 uses grouped-query attention (GQA); we expand K/V with
  ``repeat_kv`` exactly the way HF does, then call our kernel with one Q-head
  per K/V-head (a memory cost of ``n_kv_groups x`` on the expanded K/V buffers).

Both patches preserve the original module's ``forward`` / ``_attn`` as
``_orig_*`` so that incremental decoding (``q_len != kv_len`` due to a non-empty
KV cache) and any path with a custom non-causal ``attention_mask`` fall back to
the original implementation. The kernel currently only handles square self-
attention with a pure causal mask, which covers prefill.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .triton_fa import flash_attn_forward


def _is_causal_only_mask(mask: torch.Tensor, q_len: int) -> bool:
    """True if ``mask`` is shape ``(?, ?, q_len, kv_len)`` with kv_len >= q_len
    and content equal to the standard causal mask (lower + diag == 0, strict
    upper << 0). Cheap: scans only ``q_len * kv_len`` elements per call.
    """
    if mask.dim() != 4:
        return False
    if mask.size(-2) != q_len or mask.size(-1) < q_len:
        return False
    # Inspect the q_len x q_len top-left block (the only one that affects output
    # when q.shape == k.shape after projection).
    m = mask[..., :q_len, :q_len]
    # Lower triangle (incl. diag) should be exactly 0; strict upper should be
    # very negative. Use a generous threshold for fp16.
    tri = torch.tril(torch.ones(q_len, q_len, device=mask.device, dtype=torch.bool))
    if (m[..., tri].abs() > 1e-3).any():
        return False
    if (m[..., ~tri] > -1.0).any():
        return False
    return True


# ---------------------------------------------------------------------------
# GPT-2
# ---------------------------------------------------------------------------

def _flash_gpt2_attn(self, query, key, value, attention_mask=None, head_mask=None):
    """Drop-in replacement for ``GPT2Attention._attn``.

    Shapes follow HF GPT-2:
        query, key, value: (B, H, N, D)  fp16 (when the model is in fp16)
    The kernel wants ``(B, N, H, D)``.

    We only take the fast path when:
        * the layer is *not* cross-attention,
        * ``head_mask`` is ``None``,
        * ``attention_mask`` is ``None`` (i.e. no padding),
        * the sequence is square (``q_len == k_len``; prefill, no KV-cache step).
    """
    if (
        getattr(self, "is_cross_attention", False)
        or head_mask is not None
        or query.size(-2) != key.size(-2)
    ):
        return self._orig_attn(query, key, value, attention_mask, head_mask)
    if attention_mask is not None and not _is_causal_only_mask(attention_mask, query.size(-2)):
        return self._orig_attn(query, key, value, attention_mask, head_mask)

    # (B, H, N, D) -> (B, N, H, D)
    q = query.transpose(1, 2).contiguous().to(torch.float16)
    k = key.transpose(1, 2).contiguous().to(torch.float16)
    v = value.transpose(1, 2).contiguous().to(torch.float16)

    head_dim = q.size(-1)
    sm_scale = (head_dim ** -0.5)
    if getattr(self, "scale_attn_by_inverse_layer_idx", False):
        sm_scale = sm_scale / float(self.layer_idx + 1)

    out = flash_attn_forward(q, k, v, causal=True, sm_scale=sm_scale)
    out = out.to(value.dtype).transpose(1, 2).contiguous()
    return out, None


def patch_gpt2(model: torch.nn.Module) -> int:
    """Patch every GPT2Attention block on ``model``. Returns count patched."""
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

    n = 0
    for mod in model.modules():
        if isinstance(mod, GPT2Attention):
            if not hasattr(mod, "_orig_attn"):
                mod._orig_attn = mod._attn
            mod._attn = _flash_gpt2_attn.__get__(mod, type(mod))
            n += 1
    return n


def unpatch_gpt2(model: torch.nn.Module) -> int:
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

    n = 0
    for mod in model.modules():
        if isinstance(mod, GPT2Attention) and hasattr(mod, "_orig_attn"):
            mod._attn = mod._orig_attn
            del mod._orig_attn
            n += 1
    return n


# ---------------------------------------------------------------------------
# Qwen2
# ---------------------------------------------------------------------------

def _flash_qwen2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
):
    """Replacement for ``Qwen2{Sdpa|Eager|Flash2}Attention.forward``.

    Only takes the fast path when:
        * ``output_attentions`` is False,
        * ``past_key_value`` is None *or* empty for this layer (prefill),
        * ``attention_mask`` is None (no padding / no custom 4D mask),
        * the resulting sequence is square (q_len == kv_len).

    Otherwise dispatches to the original ``forward`` (stored as
    ``self._orig_forward``).
    """
    bsz, q_len, _ = hidden_states.size()

    # Cheap exit conditions -- KV cache present, output_attentions, padding mask.
    cache_has_content = False
    if past_key_value is not None:
        try:
            cache_has_content = past_key_value.get_usable_length(q_len, self.layer_idx) > 0
        except (AttributeError, TypeError):
            cache_has_content = True
    mask_ok = attention_mask is None or _is_causal_only_mask(attention_mask, q_len)
    if output_attentions or cache_has_content or not mask_ok:
        return self._orig_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # (B, N, H_total) -> (B, H, N, D)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # Update KV cache if requested. We store the *unrepeated* K/V (matches HF).
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # Expand K/V to match Q's head count (GQA -> MHA).
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # (B, H, N, D) -> (B, N, H, D), fp16, contiguous.
    q = query_states.transpose(1, 2).contiguous().to(torch.float16)
    k = key_states.transpose(1, 2).contiguous().to(torch.float16)
    v = value_states.transpose(1, 2).contiguous().to(torch.float16)

    sm_scale = self.head_dim ** -0.5
    attn_output = flash_attn_forward(q, k, v, causal=True, sm_scale=sm_scale)

    # Back to (B, N, H_total) and project.
    attn_output = attn_output.to(hidden_states.dtype).contiguous().view(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


def _is_qwen2_attention(mod) -> bool:
    cls_name = type(mod).__name__
    return cls_name in {"Qwen2Attention", "Qwen2SdpaAttention", "Qwen2FlashAttention2"}


def _is_llama_attention(mod) -> bool:
    cls_name = type(mod).__name__
    return cls_name in {"LlamaAttention", "LlamaSdpaAttention", "LlamaFlashAttention2"}


def patch_qwen2(model: torch.nn.Module) -> int:
    """Patch every Qwen2*Attention block on ``model``. Returns count patched."""
    n = 0
    for mod in model.modules():
        if _is_qwen2_attention(mod):
            if not hasattr(mod, "_orig_forward"):
                mod._orig_forward = mod.forward
            mod.forward = _flash_qwen2_forward.__get__(mod, type(mod))
            n += 1
    return n


def unpatch_qwen2(model: torch.nn.Module) -> int:
    n = 0
    for mod in model.modules():
        if _is_qwen2_attention(mod) and hasattr(mod, "_orig_forward"):
            mod.forward = mod._orig_forward
            del mod._orig_forward
            n += 1
    return n


# ---------------------------------------------------------------------------
# Auto-dispatch helper
# ---------------------------------------------------------------------------

def patch_model(model: torch.nn.Module) -> Tuple[str, int]:
    """Detect model family and patch it. Returns ``(family, n_patched)``."""
    cls = type(model).__name__
    if "GPT2" in cls or any(type(m).__name__ == "GPT2Attention" for m in model.modules()):
        return "gpt2", patch_gpt2(model)
    if "Qwen2" in cls or any(_is_qwen2_attention(m) for m in model.modules()):
        return "qwen2", patch_qwen2(model)
    if "Llama" in cls or any(_is_llama_attention(m) for m in model.modules()):
        return "llama", patch_llama(model)
    raise ValueError(f"no patch known for model class {cls}")


# ---------------------------------------------------------------------------
# Llama (alias of the Qwen2 patch -- attention shape & GQA path are identical
# at the HF module level; only the rotary helper module differs)
# ---------------------------------------------------------------------------

def _flash_llama_forward(self, *args, **kwargs):
    """Llama variant. The body is structurally identical to the Qwen2 forward
    except it imports ``apply_rotary_pos_emb`` / ``repeat_kv`` from the
    ``modeling_llama`` namespace. To keep one source of truth we call into
    ``_flash_qwen2_forward`` after monkey-importing the right helpers into
    Qwen2's namespace -- but that is fragile, so instead we just re-import
    the helpers from llama here. The body is otherwise the same."""
    # Import once at first call to avoid hard-importing llama at module load.
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    hidden_states = kwargs.get("hidden_states", args[0] if args else None)
    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)
    past_key_value = kwargs.get("past_key_value", None)
    output_attentions = kwargs.get("output_attentions", False)
    use_cache = kwargs.get("use_cache", False)
    cache_position = kwargs.get("cache_position", None)

    bsz, q_len, _ = hidden_states.size()
    cache_has_content = False
    if past_key_value is not None:
        try:
            cache_has_content = past_key_value.get_usable_length(q_len, self.layer_idx) > 0
        except (AttributeError, TypeError):
            cache_has_content = True
    mask_ok = attention_mask is None or _is_causal_only_mask(attention_mask, q_len)
    if output_attentions or cache_has_content or not mask_ok:
        return self._orig_forward(**kwargs) if kwargs else self._orig_forward(*args)

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    q = query_states.transpose(1, 2).contiguous().to(torch.float16)
    k = key_states.transpose(1, 2).contiguous().to(torch.float16)
    v = value_states.transpose(1, 2).contiguous().to(torch.float16)
    sm_scale = self.head_dim ** -0.5
    attn_output = flash_attn_forward(q, k, v, causal=True, sm_scale=sm_scale)
    attn_output = attn_output.to(hidden_states.dtype).contiguous().view(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


def patch_llama(model: torch.nn.Module) -> int:
    """Patch every Llama*Attention block on ``model``. Returns count patched.

    Untested on real Llama weights (the validation matrix uses Qwen2.5-0.5B);
    treat as a structural mirror of ``patch_qwen2`` until exercised.
    """
    n = 0
    for mod in model.modules():
        if _is_llama_attention(mod):
            if not hasattr(mod, "_orig_forward"):
                mod._orig_forward = mod.forward
            mod.forward = _flash_llama_forward.__get__(mod, type(mod))
            n += 1
    return n


def unpatch_llama(model: torch.nn.Module) -> int:
    n = 0
    for mod in model.modules():
        if _is_llama_attention(mod) and hasattr(mod, "_orig_forward"):
            mod.forward = mod._orig_forward
            del mod._orig_forward
            n += 1
    return n
