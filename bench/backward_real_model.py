"""End-to-end fwd+bwd throughput on Qwen2.5-7B with the FA-volta patch.

Measures, on a single V100 32GB at fp16:

    * tok/s for fwd-only and fwd+bwd, eager vs FA-patched, at seq ∈ {1024,
      2048, 4096}.
    * Peak GPU memory for fwd+bwd, eager vs FA-patched.
    * Maximum trainable sequence length under each implementation, found by
      doubling S until OOM (so we can quote "max trainable seq on V100 went
      from S to S' with FA backward").

Training-shape protocol (matches the parity test):

    * ``model.eval()`` — Qwen2.5 has no Dropout on the path we exercise;
      explicit grad control isolates the kernel from train-mode side effects.
    * Freeze all params, then enable grad on ``lm_head`` + first / mid / last
      ``q_proj`` weights. This forces the autograd graph to stay live through
      every layer (lm_head's backward path traverses the whole stack), so the
      attention forward must save its intermediates the same as a "real"
      training step would. The eager attention saves the (B, H, S, S) softmax
      probability tensor; the FA path doesn't — that is the headline win.
    * Loss is CE against shifted labels (``model(ids, labels=ids)``).
    * Backward pass is ``loss.backward()`` followed by zeroing grads.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from flash_attn_volta.patch_hf import patch_qwen2, unpatch_qwen2  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-7B"
SEQ_LENS = [1024, 1536, 1792, 2048, 4096]
N_WARMUP = 1
N_TIMED = 3
BATCH = 1


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, attn_implementation="eager"
    ).cuda()
    return tok, model


def _enable_train_grads(model):
    """Freeze everything, then enable grad on lm_head + first/mid/last q_proj.
    Returns the list of parameters with grads enabled."""
    for p in model.parameters():
        p.requires_grad_(False)
    layers = model.model.layers
    n_layers = len(layers)
    targets = [
        model.lm_head.weight,
        layers[0].self_attn.q_proj.weight,
        layers[n_layers // 2].self_attn.q_proj.weight,
        layers[-1].self_attn.q_proj.weight,
    ]
    for p in targets:
        p.requires_grad_(True)
    return targets


def _zero_grads(params):
    for p in params:
        if p.grad is not None:
            p.grad = None


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _time_fwd(model, ids):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model(ids, use_cache=False)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _time_fwd_bwd(model, ids, params):
    _zero_grads(params)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(ids, labels=ids, use_cache=False)
    out.loss.backward()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt, float(out.loss.item())


def _measure(fn, n_warmup=N_WARMUP, n_timed=N_TIMED):
    """Run fn() n_warmup+n_timed times. Returns (median_time_s, peak_mb)."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(n_timed):
        times.append(fn())
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    return statistics.median(times), peak_mb


# ---------------------------------------------------------------------------
# Per-seq, per-mode bench
# ---------------------------------------------------------------------------

def bench_at_seq(model, params, seq, mode):
    """mode in {"fwd_only", "fwd_bwd"}. Returns dict of metrics or None on OOM."""
    token_id = 0  # any in-vocab id
    ids = torch.full((BATCH, seq), token_id, dtype=torch.long, device="cuda")

    try:
        if mode == "fwd_only":
            t, peak = _measure(lambda: _time_fwd(model, ids))
        elif mode == "fwd_bwd":
            t, peak = _measure(lambda: _time_fwd_bwd(model, ids, params)[0])
        else:
            raise ValueError(mode)
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        gc.collect()
        return {"oom": True, "msg": str(e).split("\n", 1)[0]}

    tps = BATCH * seq / t
    return {
        "oom": False,
        "time_s": round(t, 4),
        "tok_per_s": round(tps, 1),
        "peak_mb": round(peak, 1),
    }


def bench_throughput_table(tok, model):
    """Cross product of {SEQ_LENS} x {fwd_only, fwd_bwd} x {eager, fa}.

    eager runs first per seq, then patch + fa. Always re-instantiate ids /
    clear cache between runs to keep peak measurements clean."""
    params = _enable_train_grads(model)
    rows = []

    for seq in SEQ_LENS:
        for mode in ["fwd_only", "fwd_bwd"]:
            print(f"  seq={seq:5d}  mode={mode:8s}", flush=True)
            # Eager.
            torch.cuda.empty_cache()
            gc.collect()
            r_ref = bench_at_seq(model, params, seq, mode)

            # FA.
            patch_qwen2(model)
            try:
                torch.cuda.empty_cache()
                gc.collect()
                r_fa = bench_at_seq(model, params, seq, mode)
            finally:
                unpatch_qwen2(model)

            row = {"seq": seq, "mode": mode, "ref": r_ref, "fa": r_fa}
            rows.append(row)
            ref_str = "OOM" if r_ref.get("oom") else f"{r_ref['tok_per_s']:.0f} tok/s, {r_ref['peak_mb']:.0f} MB"
            fa_str  = "OOM" if r_fa.get("oom")  else f"{r_fa['tok_per_s']:.0f} tok/s, {r_fa['peak_mb']:.0f} MB"
            print(f"      ref: {ref_str}    fa: {fa_str}", flush=True)

    return rows


# ---------------------------------------------------------------------------
# Max trainable seq (binary search by doubling)
# ---------------------------------------------------------------------------

MAX_SEQ_PROBE = [1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096, 6144, 8192]


def _try_fwd_bwd_once(model, params, seq):
    """Single fwd+bwd attempt at given seq. Returns True on success, False on OOM."""
    ids = torch.full((BATCH, seq), 0, dtype=torch.long, device="cuda")
    try:
        _zero_grads(params)
        torch.cuda.synchronize()
        out = model(ids, labels=ids, use_cache=False)
        out.loss.backward()
        torch.cuda.synchronize()
        return True
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return False
    finally:
        del ids
        _zero_grads(params)
        torch.cuda.empty_cache()
        gc.collect()


def find_max_seq(model, params, label):
    """Probe sequence lengths in MAX_SEQ_PROBE order; return the largest that
    successfully ran fwd+bwd."""
    print(f"\n=== finding max trainable seq ({label}) ===", flush=True)
    last_ok = None
    for seq in MAX_SEQ_PROBE:
        torch.cuda.empty_cache()
        gc.collect()
        ok = _try_fwd_bwd_once(model, params, seq)
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"  seq={seq:6d}  {'OK ' if ok else 'OOM'}  peak={peak_mb:.0f} MB", flush=True)
        torch.cuda.reset_peak_memory_stats()
        if ok:
            last_ok = seq
        else:
            break
    return last_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "..", "results", "backward_real_model.json"),
    )
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--skip-maxseq", action="store_true")
    args = parser.parse_args()

    print(
        f"device: {torch.cuda.get_device_name(0)}  "
        f"cap: {torch.cuda.get_device_capability(0)}",
        flush=True,
    )

    tok, model = _load_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {MODEL_ID}: {n_params/1e9:.2f}B params", flush=True)

    out = {
        "model": MODEL_ID,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "params": n_params,
        "throughput": None,
        "max_seq_eager": None,
        "max_seq_fa": None,
    }

    if not args.skip_throughput:
        print("\n=== throughput (fwd-only + fwd+bwd, eager vs FA) ===", flush=True)
        try:
            out["throughput"] = bench_throughput_table(tok, model)
        except Exception as e:
            print("throughput bench errored:")
            traceback.print_exc()
            out["throughput_error"] = str(e)

    if not args.skip_maxseq:
        params = _enable_train_grads(model)
        try:
            out["max_seq_eager"] = find_max_seq(model, params, "eager")
        except Exception as e:
            print("eager max-seq probe errored:")
            traceback.print_exc()
        patch_qwen2(model)
        try:
            out["max_seq_fa"] = find_max_seq(model, params, "FA")
        finally:
            unpatch_qwen2(model)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)

    # Markdown summary.
    if out["throughput"]:
        print("\n## fwd+bwd throughput")
        print("| seq | mode | ref tok/s | fa tok/s | speedup | ref peak MB | fa peak MB | mem saved |")
        print("|---:|---|---:|---:|---:|---:|---:|---:|")
        for r in out["throughput"]:
            ref = r["ref"]; fa = r["fa"]
            ref_t = "OOM" if ref.get("oom") else f"{ref['tok_per_s']:.0f}"
            fa_t  = "OOM" if fa.get("oom")  else f"{fa['tok_per_s']:.0f}"
            ref_m = "—"   if ref.get("oom") else f"{ref['peak_mb']:.0f}"
            fa_m  = "—"   if fa.get("oom")  else f"{fa['peak_mb']:.0f}"
            if not (ref.get("oom") or fa.get("oom")):
                speedup = f"{fa['tok_per_s'] / ref['tok_per_s']:.2f}x"
                saved   = f"{ref['peak_mb'] - fa['peak_mb']:.0f} MB"
            else:
                speedup = "—"; saved = "—"
            print(f"| {r['seq']} | {r['mode']} | {ref_t} | {fa_t} | {speedup} | {ref_m} | {fa_m} | {saved} |")

    print(f"\nmax trainable seq (eager) = {out['max_seq_eager']}")
    print(f"max trainable seq (FA)    = {out['max_seq_fa']}")


if __name__ == "__main__":
    main()
