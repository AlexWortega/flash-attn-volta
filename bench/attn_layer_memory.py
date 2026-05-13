"""Per-attention-layer peak-memory probe.

Whole-model peak memory understates the kernel's benefit because the lm_head
output ``(1, seq, vocab)`` dominates the peak. To see what FlashAttention
actually saves we extract **one** attention layer from a real model and run
just that layer on a synthetic hidden-state batch, measuring peak memory for:

  * eager-style attention (materialises the full ``(B, H, N, N)`` matrix),
  * flash_attn_volta kernel (tile-streaming, no full matrix).

Reports ``ref MB - fa MB`` -- the per-layer memory the kernel removes.
For Qwen2.5-7B (head_dim=128, n_heads=28, GQA 28:4) at seq=4096 the eager
attention matrix alone is ``28 * 4096^2 * 2 = 940 MB``.

Note: on Qwen2.5-7B the fp16 eager attention output is NaN due to QK^T
overflow in late layers (see tests/test_real_model.py). The *memory* profile
is unaffected -- NaN values still occupy fp16 storage. We are not making
correctness claims here, only measuring allocation.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from flash_attn_volta.triton_fa import flash_attn_forward  # noqa: E402

MODELS = [
    ("Qwen/Qwen2.5-0.5B", "qwen2"),
    ("Qwen/Qwen2.5-7B", "qwen2"),
]
SEQ_LENS = [1024, 2048, 4096]
DEVICE = "cuda"
DTYPE = torch.float16


def _extract_qwen2_layer(model_id: str):
    """Return ``(attn_layer, num_heads, num_kv_heads, head_dim, hidden_size)``
    for the first Qwen2 attention layer of the given model. The attention
    layer is loaded onto CUDA fp16 and put in eval mode. We use only this
    single layer to keep peak memory measurement scoped to one attention."""
    from transformers import AutoConfig, AutoModelForCausalLM

    print(f"  loading {model_id}...", flush=True)
    m = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=DTYPE, attn_implementation="eager"
    ).cuda().eval()
    attn = m.model.layers[0].self_attn
    cfg = m.config
    info = (
        attn,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.hidden_size // cfg.num_attention_heads,
        cfg.hidden_size,
    )
    return m, info


def _eager_attn_ref(attn, hidden_states):
    """Eager Qwen2-style attention: materialise full QK^T matrix. Returns
    output tensor; *intentionally* eager so the (B,H,N,N) buffer shows up in
    peak memory."""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    bsz, q_len, _ = hidden_states.size()
    q = attn.q_proj(hidden_states).view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(hidden_states).view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(hidden_states).view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)
    cos, sin = attn.rotary_emb(v, seq_len=q_len)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    k = repeat_kv(k, attn.num_key_value_groups)
    v = repeat_kv(v, attn.num_key_value_groups)

    scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(attn.head_dim)
    mask = torch.full((q_len, q_len), float("-inf"), device=scores.device, dtype=scores.dtype)
    mask = torch.triu(mask, diagonal=1)
    scores = scores + mask[None, None, :, :]
    p = torch.softmax(scores.float(), dim=-1).to(hidden_states.dtype)  # std HF practice
    out = torch.matmul(p, v)
    out = out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
    return attn.o_proj(out)


def _kernel_attn(attn, hidden_states):
    """Tile-streaming attention via flash_attn_volta."""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    bsz, q_len, _ = hidden_states.size()
    q = attn.q_proj(hidden_states).view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(hidden_states).view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(hidden_states).view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)
    cos, sin = attn.rotary_emb(v, seq_len=q_len)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    k = repeat_kv(k, attn.num_key_value_groups)
    v = repeat_kv(v, attn.num_key_value_groups)

    qb = q.transpose(1, 2).contiguous()
    kb = k.transpose(1, 2).contiguous()
    vb = v.transpose(1, 2).contiguous()
    sm_scale = attn.head_dim ** -0.5
    out = flash_attn_forward(qb, kb, vb, causal=True, sm_scale=sm_scale)
    out = out.contiguous().view(bsz, q_len, -1)
    return attn.o_proj(out)


def _peak_for(attn, seq, hidden_size, fn):
    h = torch.randn(1, seq, hidden_size, device=DEVICE, dtype=DTYPE)
    # warmup
    with torch.no_grad():
        _ = fn(attn, h)
    torch.cuda.synchronize()
    del _
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = fn(attn, h)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    del out, h
    torch.cuda.empty_cache()
    gc.collect()
    return peak


def probe_one(model_id: str):
    print(f"\n=== {model_id} ===", flush=True)
    m, (attn, n_h, n_kv, d_h, hidden) = _extract_qwen2_layer(model_id)
    print(
        f"  num_heads={n_h}  num_kv_heads={n_kv}  head_dim={d_h}  hidden={hidden}",
        flush=True,
    )

    rows = []
    for seq in SEQ_LENS:
        try:
            ref_mb = _peak_for(attn, seq, hidden, _eager_attn_ref)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  seq={seq}: ref OOM -- {e}", flush=True)
            ref_mb = float("nan")
        try:
            fa_mb = _peak_for(attn, seq, hidden, _kernel_attn)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  seq={seq}: fa OOM -- {e}", flush=True)
            fa_mb = float("nan")
        saved = ref_mb - fa_mb
        # Reference: just the (B,H,N,N) attention matrix in fp16.
        attn_matrix_mb = n_h * seq * seq * 2 / 1024 / 1024
        rows.append({
            "model": model_id,
            "seq": seq,
            "ref_peak_mb": round(ref_mb, 1) if not math.isnan(ref_mb) else None,
            "fa_peak_mb": round(fa_mb, 1) if not math.isnan(fa_mb) else None,
            "saved_mb": round(saved, 1) if not math.isnan(saved) else None,
            "attn_matrix_mb_theoretical": round(attn_matrix_mb, 1),
        })
        print(
            f"  seq={seq}: ref {ref_mb:.1f} MB  →  fa {fa_mb:.1f} MB  "
            f"saved {saved:.1f} MB  (theoretical attn-matrix = {attn_matrix_mb:.1f} MB)",
            flush=True,
        )

    del m, attn
    torch.cuda.empty_cache()
    gc.collect()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "..", "results", "attn_layer_memory.json"),
    )
    args = p.parse_args()

    print(f"device: {torch.cuda.get_device_name(0)}  cap: {torch.cuda.get_device_capability(0)}", flush=True)
    all_rows = []
    for mid, _fam in MODELS:
        try:
            all_rows.extend(probe_one(mid))
        except Exception as e:
            print(f"  ERR on {mid}: {e}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)

    print("\n| model | seq | ref MB | fa MB | saved MB | theoretical attn-matrix MB |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in all_rows:
        print(
            f"| {r['model']} | {r['seq']} | "
            f"{r['ref_peak_mb']} | {r['fa_peak_mb']} | "
            f"{r['saved_mb']} | {r['attn_matrix_mb_theoretical']} |"
        )


if __name__ == "__main__":
    main()
