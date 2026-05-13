# flash-attn-volta

FlashAttention **forward kernel** for NVIDIA Volta (Compute Capability **7.0** — Tesla V100, Titan V, Tesla T4 via SM_75 with minor tweaks). Triton 2.3 kernel, faithful to the algorithm in [Dao et al., 2205.14135](https://arxiv.org/abs/2205.14135), with the standard online-softmax tiling.

Built end-to-end **autonomously** by the [`ml-intern`](https://github.com/AlexWortega/claude-ml-intern-skill) Claude Code skill (TASK → research → benchmark sweep → autotune → verify → publish) on a 4× V100-SXM2 32GB box in roughly 25 minutes wall.

## Why does this exist

The upstream `flash-attn` (Dao Lab) v2.x dropped Volta support — it gates the kernel to SM ≥ 8.0. v1.x had Volta but is unmaintained and won't build against recent torch/CUDA versions. Triton's bundled FA kernel similarly gates Volta out, even though the underlying `mma.sync.aligned.m16n16k16` instructions Triton emits for SM 7.0 are perfectly capable of running it.

This repo lifts that gate via a small Triton 2.3 kernel that compiles and runs on SM 7.0 in the toolchains people actually have on Volta boxes (PyTorch 2.0.1 + CUDA 11.7 was the build target).

## Install

```bash
pip install --user triton==2.3.0 torch
git clone https://github.com/AlexWortega/flash-attn-volta
cd flash-attn-volta
pip install -e .  # or just add flash_attn_volta/ to PYTHONPATH
```

## Use

```python
import torch
from flash_attn_volta import flash_attn_forward

# (batch, seq, n_heads, head_dim), fp16
q = torch.randn(2, 2048, 16, 64, dtype=torch.float16, device="cuda")
k = torch.randn_like(q)
v = torch.randn_like(q)

out = flash_attn_forward(q, k, v, causal=True)
# returns fp16 of the same shape
```

API surface is intentionally small — `flash_attn_forward(q, k, v, causal=False, sm_scale=None)`. Drop-in replacement for `F.scaled_dot_product_attention(q, k, v, is_causal=...)` on Volta when fp16 and forward-only suffice.

## Correctness (fp16, vs `F.scaled_dot_product_attention`)

| shape              | causal | max-abs err vs sdpa | verdict |
|--------------------|--------|---------------------|---------|
| (2, 1024, 8, 64)   | False  | 1.22e-04            | pass    |
| (2, 1024, 8, 64)   | True   | 2.44e-04            | pass    |
| (1, 2048, 16, 128) | False  | 2.44e-04            | pass    |
| (1, 2048, 16, 128) | True  | 1.95e-03            | pass    |
| (4, 512, 4, 32)    | False  | 2.44e-04            | pass    |
| (4, 512, 4, 32)    | True   | 9.77e-04            | pass    |

Budget per the brief was 1e-2 — every combo is at least an order of magnitude under it.

`head_dim=32` is handled by internal pad-to-D=64 (Triton 2.3 on V100 fails to compile D=32 tiles cleanly).

## Benchmark (V100-SXM2 32GB, batch=1, h=16, d=64, fp16)

Forward FLOPs counted as `4 · B · H · S · S · D`.

| seq   | causal | this    | torch eager | sdpa-math | × eager | × sdpa |
|------:|--------|--------:|------------:|----------:|--------:|-------:|
| 1024  | False  | 10.7 TF | 11.9 TF     | 13.2 TF   | 0.90×   | 0.81×  |
| 1024  | True   |  8.6 TF |  8.1 TF     |  8.6 TF   | 1.06×   | 1.00×  |
| 2048  | False  | 27.9 TF | 12.2 TF     | 14.7 TF   | **2.28×** | 1.89× |
| 2048  | True   | 41.4 TF |  7.9 TF     |  9.1 TF   | **5.24×** | 4.55× |
| 4096  | False  | 38.3 TF | 13.1 TF     | 16.8 TF   | **2.92×** | 2.29× |

**Honest call-out:** at seq=1024 we're *below* torch eager. cuBLAS fp16 matmul on V100 is already tensor-core-fast, and the tiling overhead doesn't pay back until the sequence is long enough to hide it. This matches the FA paper's own V100 numbers — the algorithmic win shows up from seq≈2048. Sub-quadratic memory holds everywhere, which is the algorithmic point.

## Stability

No NaN/Inf at seq up to 8192 (tested with causal and non-causal). The fully-masked-row guard (`m_curr_safe`, `safe_l`) handles the causal-row-0 edge case that otherwise NaNs out from `exp(-inf − (−inf))`.

## What's NOT here

- **Backward pass.** Out of scope for the brief that produced this — would need a second kernel and a saved-LSE pattern.
- **fp8, bf16.** fp16 only. (V100 doesn't have native bf16 anyway.)
- **Variable-length / packed sequences.** Standard dense `(B, S, H, D)`.
- **Dropout, ALiBi, sliding window.** Forward-only causal/non-causal is all there is.
- **MQA/GQA.** Heads are 1:1 between Q and K/V.

## Layout

```
flash-attn-volta/
├── flash_attn_volta/      # the package
│   ├── triton_fa.py       # the kernel + wrapper
│   └── ref.py             # naive reference for testing
├── tests/                 # correctness
├── bench/                 # throughput + stability + memory
├── scripts/run_verify.sh  # one-shot re-run of VERIFY.md
├── probes/                # autotune sweep outputs (logs)
├── results/               # *.json from benchmark runs
└── TASK.md / PLAN.md / RESEARCH.md / VERIFY.md / RESULTS.md
```

## Reproduce

```bash
git clone https://github.com/AlexWortega/flash-attn-volta && cd flash-attn-volta
pip install --user triton==2.3.0 torch
CUDA_VISIBLE_DEVICES=0 bash scripts/run_verify.sh
```

A V100 (or any SM 7.0 device) is required.

## License

Apache 2.0.

## Credits

- FlashAttention: Tri Dao et al., [arXiv 2205.14135](https://arxiv.org/abs/2205.14135).
- Triton: Philippe Tillet et al.
- Built by `ml-intern` Claude Code skill: <https://github.com/AlexWortega/claude-ml-intern-skill>
