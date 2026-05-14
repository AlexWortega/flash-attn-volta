# PLAN — flash-attn-volta backward

## Math (FA paper §B / Algorithm 4)

Saved from forward: `O`, `LSE = m + log(l)` (rowwise log-sum-exp).
Per-row scalar `D[i] = rowsum(O[i] * dO[i])` precomputed in a tiny kernel.

For each (i,j) tile:
- `S[i,j] = Q[i] · K[j]^T * scale`  (recompute)
- `P[i,j] = exp(S[i,j] - LSE[i])`   (recompute, safe when LSE=+inf for masked rows)
- `dV[j] += P[i,j]^T · dO[i]`
- `dP[i,j] = dO[i] · V[j]^T`
- `dS[i,j] = P[i,j] * (dP[i,j] - D[i])`
- `dQ[i] += dS[i,j] · K[j] * scale`
- `dK[j] += dS[i,j]^T · Q[i] * scale`

Causal: `dS[i,j] = 0` where `i < j`. Same triangular mask as forward.

## Files

- `flash_attn_volta/triton_fa.py` — keep `flash_attn_forward` as-is (no autograd, no LSE return). Add `flash_attn_forward_lse(q,k,v,causal,sm_scale) -> (out, lse)` that runs a near-identical forward kernel but stores `lse = m + log(l)` (with `lse = +inf` for fully-masked rows so backward `exp(s-lse) → 0`).
- `flash_attn_volta/triton_fa_bwd.py` — two Triton kernels:
  - `_fa_bwd_dq_kernel_d{64,128}`: program ↔ (Q-block, batch_head). Streams K/V tiles, accumulates dQ. Reads LSE, D.
  - `_fa_bwd_dkdv_kernel_d{64,128}`: program ↔ (K-block, batch_head). Streams Q/dO tiles, accumulates dK and dV. Reads LSE, D.
  - Python-side `flash_attn_backward(do, q, k, v, o, lse, causal, sm_scale) -> (dq, dk, dv)`.
  - Computes `D = (O * dO).sum(-1).float()` in torch (cheap, B*H*N elements, no need for own kernel).
- `flash_attn_volta/autograd.py` — `class _FlashAttnFn(torch.autograd.Function)`. Forward saves `q, k, v, o, lse` + flags. Backward calls `flash_attn_backward`. User-facing `flash_attn(q, k, v, causal=False, sm_scale=None)` that wraps it.
- `flash_attn_volta/__init__.py` — also export `flash_attn`.
- `tests/test_backward.py` — finite-diff vs fp64 SDPA + autograd, gradcheck on tiny shape, causal-leak test.
- `bench/backward.py` — fwd+bwd TFLOP/s + peak memory vs torch eager.

## Tile sizes

Same as forward: `BLOCK_M = BLOCK_N = 64`, `num_warps = 4`, `num_stages = 2`. Re-use the proven Triton 2.3 + V100 SM 7.0 config (every `tl.dot` K-dim stays at 64, including the D=128 split-half pattern).

## Numerical stability

- Forward: store `lse = m + log(safe_l)` where `safe_l = max(l, eps)`; for fully-masked rows (`l == 0`) overwrite `lse = +inf` so `exp(s - lse) → 0` in backward.
- Backward: same fp32 accumulators. `D[i]` precomputed in fp32. `tl.exp` with `-inf` argument returns 0 — that handles masked rows automatically given the `lse = +inf` sentinel.

## Success criteria

- Backward max-abs err vs fp64 SDPA+autograd reference: < **2e-2** (fp16 inputs).
- `torch.autograd.gradcheck` passes on (B=1,S=16,H=2,D=32) in fp32.
- Causal-leak: perturbing `K[j]` for `j > i` does not change `dQ[i]` for early rows.
- Bench: peak memory at fwd+bwd grows linearly (not quadratically) with seq.

## Risks

1. **D=32 backward.** Forward pads D=32 → D=64. Backward must do the same and slice the gradient back. (Pad Q,K,V to D=64; run kernels at D=64; slice dQ/dK/dV back to D=32.)
2. **dV dtype mismatch on tl.dot.** P has fp32 layout; cast to fp16 before `tl.dot(p_fp16, do_fp16)` to keep mma path.
3. **Recompute P with stale LSE for masked rows.** Mitigated by `lse=+inf` sentinel.
4. **fp32 gradcheck** needs the kernel to also accept fp32 inputs (or we materialize via SDPA). Plan: gradcheck wrapper does `q,k,v.half()`, runs flash_attn, casts back to fp32; gradcheck tolerance loosened to 1e-1 since fp16 round-trip dominates. If that fails, drop the gradcheck test to a documented "not run on fp16 backward" note.

## Milestones

- [ ] `plan_ready` — this file written.
- [ ] `code_ready` — dQ kernel compiles, single-shape correctness vs fp64 ref < 2e-2.
- [ ] `train_started` — first run of `bench/backward.py`.
- [ ] `train_done` — full test_backward + bench green, VERIFY.md updated.

## Time budget (2h wall)

| min  | task |
|-----:|------|
|  10  | PLAN, forward LSE return + sanity |
|  30  | dQ kernel + dQ-only correctness |
|  30  | dK/dV kernel + full backward correctness |
|  15  | autograd.Function + finite-diff test |
|  15  | gradcheck + causal-leak + bench |
|  10  | VERIFY/README + patches |
|  10  | buffer / debugging |
