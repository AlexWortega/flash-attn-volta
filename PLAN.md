# PLAN — flash-attn-volta

## Approach

**Option 3 — Triton kernel.** Probe (`probes/probe_triton.py`) confirms Triton 2.0.0 fp16 `tl.dot` works on V100 SM 7.0. Lowest-risk path to a working forward kernel.

## Files to create

- `flash_attn_volta/__init__.py` — re-export `flash_attn_forward`.
- `flash_attn_volta/triton_fa.py` — Triton kernel + Python wrapper. Handles `head_dim ∈ {32, 64, 128}`, causal flag.
- `flash_attn_volta/ref.py` — fp32 reference for correctness checks.
- `tests/test_correctness.py` — pytest-style script: shapes `(2,1024,8,64)`, `(1,2048,16,128)`, `(4,512,4,32)`, causal & non-causal.
- `bench/bench.py` — TFLOP/s vs torch eager + sdpa-math; memory at seq=4096.
- `bench/stability.py` — large-seq NaN/Inf check.
- `scripts/run_verify.sh` — runs all of the above and writes `VERIFY.md`.

## Kernel design

- `BLOCK_M = 128` (query block), `BLOCK_N = 64` (K/V block).
- Grid: `(ceil(seq / BLOCK_M), batch * n_heads)`.
- One program loops over K/V blocks for its query block.
- Online softmax with running `m_i` (max) and `l_i` (sum-exp), `O_i` accumulator in fp32.
- Causal mask: only K block indices `j*BLOCK_N <= i*BLOCK_M + BLOCK_M - 1` participate; partial blocks at the diagonal apply elementwise mask.
- `scale = 1/sqrt(head_dim)` baked into `tl.dot` output before subtract-max.
- For head_dim 128, layout uses `BLOCK_DMODEL = 128` (or 32/64 for the smaller cases).

## Success criteria

- Correctness: max-abs err < **1e-2** vs `F.scaled_dot_product_attention` (fp16 ref) for all six (shape × causal) combos.
- Stability: seq=4096, h=16, d=64 → no NaN/Inf, finite output.
- Speed:
  - ≥ **2× faster** than torch eager `(Q @ K.T).softmax() @ V` for seq≥1024.
  - within **50%** of `sdpa(math backend)` for seq≥1024.
- Memory: peak HBM at seq=4096, h=16, d=64 sub-quadratic in `seq` (i.e. close to O(b·h·s·d), not O(b·h·s²)).
- SM 7.0 verified via `torch.cuda.get_device_capability()`.

## Hyperparameters

| param        | value     |
|--------------|-----------|
| BLOCK_M      | 128       |
| BLOCK_N      | 64        |
| num_warps    | 4 (auto)  |
| num_stages   | 2         |
| acc dtype    | fp32      |
| io dtype     | fp16      |
| scale        | 1/√d      |

## Risks & fallbacks

1. **Triton compile fails for head_dim=128.** → drop to BLOCK_M=64.
2. **Numerical drift > 1e-2.** → switch to `exp2 / log2(e)` scaling for stable softmax.
3. **Slower than torch eager.** → tune `num_warps` to 8; try BLOCK_N=128.
4. **Triton totally unusable.** → fallback to CUDA C++ extension using `nvcuda::wmma` (file already scaffolded as backup).

## Milestones

- [ ] `code_ready` after kernel compiles + runs on a random forward.
- [ ] `train_started` (== "first benchmark") after correctness passes.
- [ ] `train_done` after `VERIFY.md` all-pass.
- [ ] `published` after HF or gist push.
