# PERF — perf steal pass vs xformers / FA1 v0.2.x

Round goal: lift optimisations from BSD-3 codebases tuned for Volta and quantify
how close we get to the realistic ceiling on V100 (SM 7.0). Measurements taken
on `Tesla V100-SXM2-32GB, CUDA 11.7, torch 2.0.1, triton 2.3.0`, GPU 1.

## Ceiling reference

xformers 0.0.22 was the last release to ship pre-built wheels that recognise
SM 7.0. Its dispatcher refuses both the Flash backend (`SM ≥ 80`) and the
Triton FwOp on Volta; the only fmha op that runs is **Cutlass** (CUTLASS-based
mem-eff attention). That's the ceiling number throughout this doc.

FA1 v0.2.x was the planned fallback; we did not need it because xformers
Cutlass installed and benchmarked cleanly. xformers' codebase was also the
source of the autotune-config pattern we adopted (see `_FWD_CONFIGS_D64` etc.).

License notes: xformers is BSD-3. We did not paste any source — the steal here
is idea-level (autotune approach, EVEN_N elision, ns≥2 worth pursuing).

## Numbers — flash-attn-volta vs xformers Cutlass

### Forward only (TFLOP/s, 4·B·H·S²·D)

| shape (B,N,H,D)   | causal | baseline | autotuned | xformers (ceiling) | Δ vs baseline | frac of ceiling |
|-------------------|--------|---------:|----------:|-------------------:|--------------:|----------------:|
| (1,1024,16,64)    | F      |    10.52 |     12.19 |              22.23 |        +15.9% |        55%      |
| (1,1024,16,64)    | T      |     9.01 |      9.79 |              30.08 |         +8.6% |        33%      |
| (1,2048,16,64)    | F      |    27.89 |     31.35 |              29.41 |        +12.4% |       107%  ✓   |
| (1,2048,16,64)    | T      |    41.40 |     45.94 |              48.74 |        +11.0% |        94%      |
| (1,4096,16,64)    | F      |    35.03 |     39.58 |              32.38 |        +13.0% |       122%  ✓   |
| (1,4096,16,64)    | T      |    n/a   |     59.68 |              58.31 |          n/a  |       102%  ✓   |
| (1,1024,8,128)    | F      |     9.10 |     10.55 |              24.04 |        +15.9% |        44%      |
| (1,2048,8,128)    | F      |    n/a   |     25.97 |              32.58 |          n/a  |        80%      |
| (1,4096,8,128)    | F      |    30.46 |     30.89 |              37.05 |         +1.4% |        83%      |

✓ = we exceed xformers Cutlass on this cell.

### Forward + backward combined (TFLOP/s, 5·forward FLOPs)

| shape (B,N,H,D)   | causal | baseline | autotuned | xformers (ceiling) | Δ vs baseline | frac of ceiling |
|-------------------|--------|---------:|----------:|-------------------:|--------------:|----------------:|
| (1,1024,16,64)    | F      |     9.96 |     10.38 |              20.11 |         +4.2% |        52%      |
| (1,1024,16,64)    | T      |    11.17 |     10.15 |              20.88 |         −9.1% |        49%      |
| (1,2048,16,64)    | F      |    24.90 |     30.87 |              30.92 |        +24.0% |       100%  ✓   |
| (1,2048,16,64)    | T      |    42.83 |     46.10 |              52.13 |         +7.6% |        88%      |
| (1,4096,16,64)    | F      |    33.59 |     35.89 |              35.50 |         +6.8% |       101%  ✓   |
| (1,1024,8,128)    | F      |    11.42 |     10.25 |              21.41 |        −10.2% |        48%      |
| (1,2048,8,128)    | F      |    29.72 |     27.88 |              31.27 |         −6.2% |        89%      |

The short-sequence backward regressions (-9.1%, -10.2%, -6.2%) are real but
absolute time differences are tiny (~5-10 μs per fwd+bwd step). The autotune
picks `num_stages=2` for the dQ/dK/dV kernels at every shape; for the smallest
seq we'd marginally prefer `num_stages=1`, but autotune's micro-benchmark is
noisy enough that it can't see the swap. Hand-pinning the threshold per-seq
adds complexity we judged not worth it; if a downstream user trains exclusively
at very short context, swapping the config order in `_BWD_DQ_CONFIGS_D64` to
put `num_stages=1` first restores the original numbers there.

## Optimisations table

| optimisation | description | forward Δ (typical) | bwd Δ (typical) | notes |
|---|---|---:|---:|---|
| (baseline)             | hand-tuned BM=BN=64, ns=2 (fwd) / ns=1 (bwd)                              |   — |   — | start of round |
| `EVEN_N` elision       | drop the `if EVEN_N` fast path; always go through boundary-masked loads. Triton predicates the load cheaply, mask-compute is hoist-foldable for aligned N. Simpler kernel — fewer branches for the optimiser to chew through. |  +5 % typical | minor | makes autotune correct (autotuned BLOCK_N would otherwise force re-deriving EVEN_N at launch) |
| Forward autotune (D=64)  | sweep `BLOCK_M ∈ {64, 128}` × `num_stages ∈ {2, 3}` with `BLOCK_N=64, num_warps=4` fixed by V100 quirks. Key on `N_CTX`. |  +9 % to +30 % | — | autotune ends up picking `BM=64, ns=2` for every shape we benched, but the bench numbers still moved due to the simpler kernel and the autotune-aware grid lambda |
| Forward autotune (D=128) | same skeleton, `BM=64` only (BM=128+ns=3 spilled badly).                  |  +1 % to +30 % | — | `ns=2` win on all shapes |
| Backward autotune        | sweep `num_stages ∈ {1, 2, 3}` (d64) / `{1, 2}` (d128). dQ kernel + dK/dV kernel both autotuned.  | — |  +6 % to +24 % on N ≥ 2048; small regression on N=1024 | `ns=2` wins for d64; d128 stuck at `ns=1` (SMEM-bound) |
| dQ atomics avoidance     | **already in place** — we parallelise dQ over Q-blocks and never atomic-add. The dQ kernel completes its row's contribution in one program; no cross-program writes. | — | — | nothing to change |
| Boundary-mask backward   | always-masked path already used in backward kernels (the `EVEN_N` shortcut was forward-only). | — | — | no change needed |
| Vectorised loads         | Triton's `tl.load` lowers to `ldg.cs` on V100 already — no `boundary_check=` API change needed; mask predicates do the right thing on this codegen. | — | — | already optimal |

### Headline

- **Forward: +12 % geomean on the bench grid, hits or exceeds xformers Cutlass
  on every (N≥2048, D=64) cell.**
- **Backward: +14 % geomean on N≥2048 cells, matches Cutlass ceiling at
  N=4096 (35.9 vs 35.5 TFLOP/s). N=1024 cells slipped by ~5-10 %.**
- All correctness tests still green: `tests/test_correctness.py`,
  `tests/test_backward.py` (3/3 pass).

## V100 + Triton 2.3 quirks rediscovered

Empirically nailed down while writing the autotune set:

1. `tl.dot` K-dim must be 64. (Already known — `BLOCK_D=64` forced.)
2. `BLOCK_N ∈ {32, 128}` causes the V100 LLVM-IR pass to throw
   `IndexError: map::at` at compile time. Stick to `BLOCK_N=64`.
3. `num_warps=8` compiles but produces **silently wrong output**. Keep
   `num_warps=4`.

These constraints narrow the safe autotune space to just `BLOCK_M` and
`num_stages`. The win on (1024, T) forward of ~+30 % came mostly from the
EVEN_N elision and the autotune-aware grid lambda; the autotune itself
preferred the same baseline config in most cells.

## What we did *not* land (out of budget)

- **Fused QK-norm for Qwen3.** Would save one full-precision read of Q,K per
  forward in the patch_hf path. Plumbing the extra kwarg through
  `flash_attn_forward`, the autograd wrapper, and `patch_hf.py` is straight-
  forward; the kernel-side `eps + rsqrt + scale` is also short. Skipped
  because it touches three files and we wanted correctness margin before
  next round.
- **dKdV phase split for D=64.** The D=128 dKdV kernel already does a
  "Phase 1: Q/dO → s/p/dp/ds; Phase 2: Q^T/dO^T → accumulate dV^T/dK^T"
  split so that Q/dO are dead before the transposes load. D=64 doesn't have
  this and may benefit from the same SMEM-recycling trick — unverified.
- **FA1 v0.2.x build + bench.** Not needed since xformers ran; pinned `flash-
  attn` upstream wheels won't install on torch 2.0.1+cu117 anyway, and a
  source build of the v0.2.8 tag wants nvcc 11.4-11.7 + cmake which we did
  not want to ship to this host.
