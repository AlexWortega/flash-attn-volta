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

from .autograd import flash_attn


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

    out = flash_attn(q, k, v, causal=True, sm_scale=sm_scale)
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
    attn_output = flash_attn(q, k, v, causal=True, sm_scale=sm_scale)

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
# Qwen3 (structurally identical attention to Qwen2 at the HF module level --
# GQA + RoPE + grouped K/V. Adds QK-norm but the projections/shapes are the
# same, so the same forward body works once the rotary helpers are imported
# from the qwen3 namespace.)
# ---------------------------------------------------------------------------

def _is_qwen3_attention(mod) -> bool:
    cls_name = type(mod).__name__
    return cls_name in {"Qwen3Attention", "Qwen3SdpaAttention", "Qwen3FlashAttention2"}


def _flash_qwen3_forward(self, *args, **kwargs):
    """Replacement for ``Qwen3{Sdpa|Eager}Attention.forward``.

    Qwen3's attention applies QK-norm (an extra RMSNorm on Q and K) before the
    rotary embedding. That happens *inside* the original forward and is not
    visible at the I/O surface — but importantly, the projection / GQA shape
    is identical to Qwen2. So rather than re-implement QK-norm we delegate
    cases where the original forward is needed (cache, custom mask, attn out)
    and otherwise patch through.

    Since QK-norm is the architectural delta, we must apply it ourselves
    before calling the kernel. We read ``self.q_norm`` / ``self.k_norm`` if
    present (Qwen3 always has them).
    """
    # Normalize args/kwargs.
    hidden_states = kwargs.get("hidden_states", args[0] if args else None)
    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)
    past_key_value = kwargs.get("past_key_value", None)
    output_attentions = kwargs.get("output_attentions", False)
    use_cache = kwargs.get("use_cache", False)
    cache_position = kwargs.get("cache_position", None)
    position_embeddings = kwargs.get("position_embeddings", None)

    bsz, q_len, _ = hidden_states.size()

    cache_has_content = False
    if past_key_value is not None:
        try:
            cache_has_content = past_key_value.get_usable_length(q_len, self.layer_idx) > 0
        except (AttributeError, TypeError):
            cache_has_content = True
    mask_ok = attention_mask is None or _is_causal_only_mask(attention_mask, q_len)
    if output_attentions or cache_has_content or not mask_ok:
        return self._orig_forward(*args, **kwargs)

    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv  # type: ignore

    q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # QK-norm (architectural delta vs Qwen2).
    if hasattr(self, "q_norm"):
        q = self.q_norm(q)
    if hasattr(self, "k_norm"):
        k = self.k_norm(k)

    # Rotary: Qwen3 may pass position_embeddings precomputed (newer path) or
    # expose a rotary_emb on the layer (older path).
    if position_embeddings is not None:
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    else:
        cos, sin = self.rotary_emb(v, seq_len=q_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

    k = repeat_kv(k, self.num_key_value_groups)
    v = repeat_kv(v, self.num_key_value_groups)

    qb = q.transpose(1, 2).contiguous().to(torch.float16)
    kb = k.transpose(1, 2).contiguous().to(torch.float16)
    vb = v.transpose(1, 2).contiguous().to(torch.float16)

    sm_scale = self.head_dim ** -0.5
    out = flash_attn(qb, kb, vb, causal=True, sm_scale=sm_scale)
    out = out.to(hidden_states.dtype).contiguous().view(bsz, q_len, -1)
    out = self.o_proj(out)
    return out, None, past_key_value


def patch_qwen3(model: torch.nn.Module) -> int:
    """Patch every Qwen3*Attention block on ``model``. Returns count patched.

    No-op if the installed ``transformers`` does not define Qwen3 (returns 0).
    """
    try:
        from transformers.models.qwen3 import modeling_qwen3  # noqa: F401
    except ImportError:
        return 0
    n = 0
    for mod in model.modules():
        if _is_qwen3_attention(mod):
            if not hasattr(mod, "_orig_forward"):
                mod._orig_forward = mod.forward
            mod.forward = _flash_qwen3_forward.__get__(mod, type(mod))
            n += 1
    return n


def unpatch_qwen3(model: torch.nn.Module) -> int:
    n = 0
    for mod in model.modules():
        if _is_qwen3_attention(mod) and hasattr(mod, "_orig_forward"):
            mod.forward = mod._orig_forward
            del mod._orig_forward
            n += 1
    return n


# ---------------------------------------------------------------------------
# Linear-attention / state-space detection (Mamba, RWKV, ...).
#
# The kernel only implements softmax (causal) attention. If someone tries to
# patch a Mamba or RWKV model we refuse loudly rather than silently produce
# garbage. ``patch_model`` raises; the dedicated ``patch_<x>`` family helpers
# above remain no-ops on non-matching modules (the safest fallthrough -- the
# model just keeps using its own native forward).
# ---------------------------------------------------------------------------

_LINEAR_ATTN_CLASS_NAMES = {
    # Mamba 1
    "MambaMixer", "MambaModel", "MambaForCausalLM",
    # Mamba 2
    "Mamba2Mixer", "Mamba2Model", "Mamba2ForCausalLM",
    # RWKV
    "RwkvSelfAttention", "RwkvLinearAttention", "RwkvModel", "RwkvForCausalLM",
    # RecurrentGemma (Griffin variant)
    "RecurrentGemmaRecurrentBlock", "RecurrentGemmaModel", "RecurrentGemmaForCausalLM",
    # RetNet (if/when it lands)
    "RetNetAttention", "RetNetModel", "RetNetForCausalLM",
}


def _linear_attn_family(model: torch.nn.Module) -> Optional[str]:
    """Return the family name (mamba / mamba2 / rwkv / ...) if the model is
    a linear-attention / recurrent / state-space architecture; else None."""
    seen_modules = {type(m).__name__ for m in model.modules()}
    seen_modules.add(type(model).__name__)
    if "MambaMixer" in seen_modules:
        return "mamba"
    if "Mamba2Mixer" in seen_modules:
        return "mamba2"
    if "RwkvSelfAttention" in seen_modules or "RwkvLinearAttention" in seen_modules:
        return "rwkv"
    if "RecurrentGemmaRecurrentBlock" in seen_modules:
        return "recurrent_gemma"
    if "RetNetAttention" in seen_modules:
        return "retnet"
    return None


# ---------------------------------------------------------------------------
# Auto-dispatch helper
# ---------------------------------------------------------------------------

def patch_model(model: torch.nn.Module) -> Tuple[str, int]:
    """Detect model family and patch it. Returns ``(family, n_patched)``.

    Refuses (with ``RuntimeError``) for any architecture that doesn't use
    softmax attention -- Mamba, RWKV, RecurrentGemma, RetNet. The kernel
    is mathematically wrong for those layers; failing fast is preferable to
    silently producing garbage logits.
    """
    fam = _linear_attn_family(model)
    if fam is not None:
        raise RuntimeError(
            f"flash-attn-volta only supports softmax MHA / GQA — model uses {fam}. "
            "Refusing to patch a linear-attention / state-space architecture."
        )
    cls = type(model).__name__
    if "GPT2" in cls or any(type(m).__name__ == "GPT2Attention" for m in model.modules()):
        return "gpt2", patch_gpt2(model)
    if "Qwen3" in cls or any(_is_qwen3_attention(m) for m in model.modules()):
        return "qwen3", patch_qwen3(model)
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
    attn_output = flash_attn(q, k, v, causal=True, sm_scale=sm_scale)
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
