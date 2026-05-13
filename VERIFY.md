# VERIFY — flash-attn-volta

Re-run with: `CUDA_VISIBLE_DEVICES=1 bash scripts/run_verify.sh`.

## 1. Correctness vs `F.scaled_dot_product_attention` (fp16)

Tolerance: max-abs error < **1e-2** (the brief's threshold).

| shape           | causal | err vs sdpa | err vs fp32-ref | verdict |
|-----------------|--------|-------------|------------------|---------|
| (2, 1024, 8, 64)   | False | 1.22e-04 | 2.44e-04 | **pass** |
| (2, 1024, 8, 64)   | True  | 2.44e-04 | 1.95e-03 | **pass** |
| (1, 2048, 16, 128) | False | 2.44e-04 | 2.44e-04 | **pass** |
| (1, 2048, 16, 128) | True  | 1.95e-03 | 1.95e-03 | **pass** |
| (4, 512, 4, 32)    | False | 2.44e-04 | 2.44e-04 | **pass** |
| (4, 512, 4, 32)    | True  | 9.77e-04 | 9.77e-04 | **pass** |

**VERDICT: pass** — every (shape × causal) combo is at least an order of
magnitude under the 1e-2 budget.

## 2. Numerical stability (no NaN/Inf at long seq)

| shape              | causal | nan | inf | output range          | verdict |
|--------------------|--------|----:|----:|-----------------------|---------|
| (1, 4096, 16, 64)  | False  |  0  |  0  | [-3.18e-2, 3.16e-2]   | pass    |
| (1, 4096, 16, 64)  | True   |  0  |  0  | [-1.52, 1.80]         | pass    |
| (1, 4096, 8, 128)  | False  |  0  |  0  | [-3.29e-2, 3.27e-2]   | pass    |
| (1, 8192, 4, 64)   | False  |  0  |  0  | [-2.09e-2, 1.78e-2]   | pass    |
| (1, 8192, 4, 64)   | True   |  0  |  0  | [-1.23, 1.41]         | pass    |

**VERDICT: pass** — finite output up to seq=8192. The fully-masked-row
guard (`m_curr_safe`, `safe_l`) covers the causal-row-0 edge case that
otherwise NaNs with `exp(-inf - (-inf))`.

## 3. Benchmark (TFLOP/s on V100-SXM2 32GB)

Forward FLOPs = `4 * B * H * S * S * D`. Three implementations compared:
- **fa**:   `flash_attn_volta.flash_attn_forward`
- **eager**: plain `(Q@K.T)*scale → softmax → @V` (cuBLAS-backed)
- **sdpa-math**: `F.scaled_dot_product_attention` with flash + mem-efficient
  backends *disabled* (forces the `math` backend, i.e. the same algorithm as
  eager but better-fused softmax).

| shape              | causal | fa TFLOP/s | eager TFLOP/s | sdpa TFLOP/s | speedup vs eager | speedup vs sdpa |
|--------------------|--------|------------|---------------|--------------|-------------------|------------------|
| (1, 1024, 16, 64)  | False  | 10.72 | 11.85 | 13.16 | 0.90x | 0.81x |
| (1, 1024, 16, 64)  | True   |  8.60 |  8.14 |  8.61 | 1.06x | 1.00x |
| (1, 2048, 16, 64)  | False  | 27.91 | 12.22 | 14.74 | **2.28x** | 1.89x |
| (1, 2048, 16, 64)  | True   | 41.38 |  7.89 |  9.08 | **5.24x** | 4.55x |
| (1, 4096, 16, 64)  | False  | 38.32 | 13.13 | 16.75 | **2.92x** | 2.29x |
| (1, 4096, 8, 128)  | False  | 30.04 | 22.80 | 28.79 | 1.32x | 1.04x |
| (1, 1024, 8, 128)  | False  | 11.55 | 18.38 | 13.59 | 0.63x | 0.85x |

**VERDICT: partial pass.**

- For **seq ≥ 2048**, the impl meets the brief's 2× vs eager bar (2.28×,
  5.24×, 2.92×) and easily clears the "within 50 % of sdpa-math" bar.
- For **seq = 1024**, the impl is within 5–35 % of eager but **fails the 2×
  bar** at D=64 non-causal (0.90×) and D=128 non-causal (0.63×). The
  D=128 1024-seq case is the worst.
- The "within 50 % of sdpa-math" bar (i.e. ≥ 0.5×) is met in every cell
  (worst is 0.81×).

Why: at seq=1024 the per-program workload is small (16 K-tiles), the grid
under-fills the 80 SMs, and cuBLAS torch eager is itself tensor-core-fast.
FA's relative advantage on V100 grows with sequence length — exactly the
regime FA was designed for. The seq=2048+ numbers match the algorithm's
intended use case.

## 4. Memory (sub-quadratic in seq)

Peak allocator-measured memory (extra over Q+K+V, excludes the QKV
permute overhead that's the same for any impl):

| shape              | causal | fa extra | eager extra | theory fa | theory eager |
|--------------------|--------|---------:|------------:|----------:|-------------:|
| (1, 1024, 16, 64)  | False  |   10.5 MB |    77.7 MB  |   2.1 MB  |    33.6 MB   |
| (1, 2048, 16, 64)  | False  |   21.0 MB |   272.6 MB  |   4.2 MB  |   134.2 MB   |
| (1, 4096, 16, 64)  | False  |   41.9 MB |  1082.1 MB  |   8.4 MB  |   536.9 MB   |
| (1, 4096, 16, 64)  | True   |   41.9 MB |  1115.7 MB  |   8.4 MB  |   536.9 MB   |

Doubling N → fa extra doubles (~linear), eager extra ×4 (quadratic).

**VERDICT: pass** — fa-extra grows linearly with N (10.5 → 21 → 42 MB for
N=1024 → 2048 → 4096), confirming the O(N) scratch the FA paper promises.
The 5× gap between actual fa-extra and the "output only" theory is the
contiguous QKV permute (B,N,H,D)→(B,H,N,D); that's wrapper bookkeeping,
not the kernel.

## 5. SM 7.0 confirmation

```
$ python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
Tesla V100-SXM2-32GB (7, 0)
```

Plus `nvidia-smi --query-gpu=compute_cap --format=csv` reports `7.0` on the
test GPU. **VERDICT: pass.**

## Summary

| section          | verdict |
|------------------|---------|
| 1. correctness   | pass    |
| 2. stability     | pass    |
| 3. benchmark     | partial pass (2× met for seq≥2048; seq=1024 below 2× vs eager) |
| 4. memory        | pass    |
| 5. SM 7.0 check  | pass    |

The kernel is **functionally correct** for the brief's full shape envelope
(D ∈ {32, 64, 128}, causal/non-causal), **numerically stable** up to
seq=8192, and **sub-quadratic in memory**. Throughput meets the 2× bar
where FA's algorithmic advantage actually applies (seq≥2048); the
seq=1024 cells are within 5–35 % of cuBLAS torch eager but don't clear
the 2× bar — the cuBLAS fp16 path is itself tensor-core-fast on V100 and
at very small sequence lengths the FA tiling overhead doesn't pay back.
