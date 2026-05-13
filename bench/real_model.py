"""Throughput + peak-memory benchmark for the HF patch.

For each model in MODELS and each sequence length in SEQ_LENS, we run a forward
pass and report:
  * tokens / second  (= batch * seq / wall_clock)
  * peak GPU memory  (MB, via torch.cuda.max_memory_allocated)

Both **unpatched** (eager HF attention) and **patched** (flash_attn_volta)
configurations are timed. Two warmup passes precede 5 timed passes; the median
of the 5 is reported. Results are written to results/real_model_bench.json and
printed as a markdown table.

Causal mask is enforced. The kernel only handles the prefill shape (q_len ==
k_len), so we do *one* forward pass per timing -- not generate(). This is what
matters for context ingestion throughput.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from flash_attn_volta.patch_hf import (  # noqa: E402
    patch_gpt2,
    patch_qwen2,
    patch_qwen3,
    unpatch_gpt2,
    unpatch_qwen2,
    unpatch_qwen3,
)

MODELS = [
    # (model_id, family, max_seq_supported)
    ("gpt2", "gpt2", 1024),                       # gpt2 has hard 1024 context limit
    ("Qwen/Qwen2.5-0.5B", "qwen2", 32768),
    ("Qwen/Qwen2.5-7B", "qwen2", 32768),
]
SEQ_LENS = [1024, 2048, 4096]
N_WARMUP = 2
N_TIMED = 5
BATCH = 1


def _patch_unpatch(family: str):
    if family == "gpt2":
        return patch_gpt2, unpatch_gpt2
    if family == "qwen2":
        return patch_qwen2, unpatch_qwen2
    if family == "qwen3":
        return patch_qwen3, unpatch_qwen3
    raise ValueError(family)


def _time_forward(model, input_ids):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model(input_ids, use_cache=False)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _measure(model, input_ids):
    times = []
    for _ in range(N_WARMUP):
        _time_forward(model, input_ids)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(N_TIMED):
        times.append(_time_forward(model, input_ids))
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    return statistics.median(times), peak_mb


def bench_one(model_id: str, family: str, max_seq: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n=== {model_id} ({family}) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).cuda().eval()

    rows = []
    for seq in SEQ_LENS:
        if seq > max_seq:
            print(f"  seq={seq}: SKIP (model max_seq={max_seq})", flush=True)
            continue

        # Use repeated token ids; content doesn't matter for timing.
        token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
        input_ids = torch.full((BATCH, seq), token_id, dtype=torch.long, device="cuda")

        # Unpatched baseline.
        torch.cuda.empty_cache()
        gc.collect()
        t_ref, mem_ref = _measure(model, input_ids)

        # Patch + measure.
        patch, unpatch = _patch_unpatch(family)
        patch(model)
        try:
            torch.cuda.empty_cache()
            gc.collect()
            t_fa, mem_fa = _measure(model, input_ids)
        finally:
            unpatch(model)

        tps_ref = BATCH * seq / t_ref
        tps_fa = BATCH * seq / t_fa
        speedup = tps_fa / tps_ref
        mem_ratio = mem_fa / mem_ref

        row = {
            "model": model_id,
            "family": family,
            "seq": seq,
            "batch": BATCH,
            "time_ref_s": round(t_ref, 4),
            "time_fa_s": round(t_fa, 4),
            "tok_per_s_ref": round(tps_ref, 1),
            "tok_per_s_fa": round(tps_fa, 1),
            "speedup_x": round(speedup, 3),
            "mem_ref_mb": round(mem_ref, 1),
            "mem_fa_mb": round(mem_fa, 1),
            "mem_ratio": round(mem_ratio, 3),
        }
        rows.append(row)
        print(
            f"  seq={seq}: ref {t_ref*1000:.1f} ms ({tps_ref:.0f} tok/s, "
            f"{mem_ref:.1f} MB)  →  fa {t_fa*1000:.1f} ms ({tps_fa:.0f} tok/s, "
            f"{mem_fa:.1f} MB)  speedup {speedup:.2f}x  mem {mem_ratio:.2f}x",
            flush=True,
        )

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "..", "results", "real_model_bench.json"),
    )
    args = parser.parse_args()

    print(
        f"device: {torch.cuda.get_device_name(0)}  "
        f"cap: {torch.cuda.get_device_capability(0)}",
        flush=True,
    )
    all_rows = []
    for model_id, family, max_seq in MODELS:
        try:
            all_rows.extend(bench_one(model_id, family, max_seq))
        except Exception as e:
            print(f"  ERR: {e}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)

    # Print markdown table.
    print("\n| model | seq | ref tok/s | fa tok/s | speedup | ref MB | fa MB | mem ratio |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in all_rows:
        print(
            f"| {r['model']} | {r['seq']} | {r['tok_per_s_ref']:.0f} | "
            f"{r['tok_per_s_fa']:.0f} | {r['speedup_x']:.2f}x | "
            f"{r['mem_ref_mb']:.0f} | {r['mem_fa_mb']:.0f} | {r['mem_ratio']:.2f}x |"
        )


if __name__ == "__main__":
    main()
