# flash-attn-volta

FlashAttention **forward + backward kernels** for NVIDIA Volta (Compute Capability **7.0** — Tesla V100, Titan V, Tesla T4 via SM_75 with minor tweaks). Triton 2.3 kernels, faithful to the algorithm in [Dao et al., 2205.14135](https://arxiv.org/abs/2205.14135), with the standard online-softmax tiling on the forward and Algorithm 4 on the backward. Triton-autotuned tile/`num_stages` after a perf-steal pass against xformers Cutlass.

Built end-to-end **autonomously** by the [`ml-intern`](https://github.com/AlexWortega/claude-ml-intern-skill) Claude Code skill (TASK → research → benchmark sweep → autotune → real-model validation → perf-steal → verify → publish) across several rounds on a 4× V100-SXM2 32GB box.

## Why does this exist

The upstream `flash-attn` (Dao Lab) v2.x dropped Volta support — it gates the kernel to SM ≥ 8.0. v1.x had Volta but is unmaintained and won't build against recent torch/CUDA versions. Triton's bundled FA kernel similarly gates Volta out, even though the underlying `mma.sync.aligned.m16n16k16` instructions Triton emits for SM 7.0 are perfectly capable of running it. xformers' Flash/Triton backends also refuse SM<8.0; only their Cutlass backend runs on Volta.

This repo lifts the gate via a small Triton 2.3 kernel that compiles and runs on SM 7.0 in the toolchains people actually have on Volta boxes (PyTorch 2.0.1 + CUDA 11.7 was the build target).

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
from flash_attn_volta import flash_attn, flash_attn_forward

# (batch, seq, n_heads, head_dim), fp16
q = torch.randn(2, 2048, 16, 64, dtype=torch.float16, device="cuda", requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)

# Autograd-aware (forward + backward via the Triton kernels).
out  = flash_attn(q, k, v, causal=True)
loss = out.sum()
loss.backward()    # populates q.grad, k.grad, v.grad

# Or the raw forward (no autograd, no LSE saved) for inference paths:
out_inf = flash_attn_forward(q.detach(), k.detach(), v.detach(), causal=True)
```

API surface — `flash_attn(q, k, v, causal=False, sm_scale=None)` for training, `flash_attn_forward(...)` (with optional `return_softmax_lse=True`) for inference. Drop-in replacement for `F.scaled_dot_product_attention(q, k, v, is_causal=...)` on Volta when fp16 suffices.

## Correctness (fp16, vs `F.scaled_dot_product_attention`)

| shape              | causal | max-abs err vs sdpa | verdict |
|--------------------|--------|---------------------|---------|
| (2, 1024, 8, 64)   | False  | 1.22e-04            | pass    |
| (2, 1024, 8, 64)   | True   | 2.44e-04            | pass    |
| (1, 2048, 16, 128) | False  | 2.44e-04            | pass    |
| (1, 2048, 16, 128) | True   | 1.95e-03            | pass    |
| (4, 512, 4, 32)    | False  | 2.44e-04            | pass    |
| (4, 512, 4, 32)    | True   | 9.77e-04            | pass    |

Budget per the brief was 1e-2 — every combo is at least an order of magnitude under it. `head_dim=32` is handled by internal pad-to-D=64 (Triton 2.3 on V100 fails to compile D=32 tiles cleanly).

## Forward benchmark (V100-SXM2 32GB, fp16, post-autotune)

Forward FLOPs counted as `4 · B · H · S · S · D`. **TFLOP/s**.

| shape (B,N,H,D)   | causal | this kernel | torch eager | xformers Cutlass | × eager | frac of xformers |
|-------------------|--------|------------:|------------:|-----------------:|--------:|-----------------:|
| (1,1024,16,64)    | F      |     12.2 | 11.9 |  22.2 | 1.03× | 55% |
| (1,1024,16,64)    | T      |      9.8 |  8.1 |  30.1 | 1.21× | 33% |
| (1,2048,16,64)    | F      |     31.4 | 12.2 |  29.4 | **2.57×** | **107%** ✓ |
| (1,2048,16,64)    | T      |     45.9 |  7.9 |  48.7 | **5.81×** | 94% |
| (1,4096,16,64)    | F      |     39.6 | 13.1 |  32.4 | **3.02×** | **122%** ✓ |
| (1,4096,16,64)    | T      |     59.7 |  n/a |  58.3 | — | **102%** ✓ |
| (1,1024, 8,128)   | F      |     10.6 |  n/a |  24.0 | — | 44% |
| (1,2048, 8,128)   | F      |     26.0 |  n/a |  32.6 | — | 80% |
| (1,4096, 8,128)   | F      |     30.9 |  n/a |  37.1 | — | 83% |

✓ = exceeds xformers Cutlass (the best V100-capable ceiling we measured).

**Forward+backward combined** (TFLOP/s, 5·forward FLOPs — bwd ≈ 4·fwd per FA paper):

| shape (B,N,H,D)   | causal | this kernel | xformers Cutlass | frac of xformers | fa peak mem | eager peak mem |
|-------------------|--------|------------:|-----------------:|-----------------:|------------:|---------------:|
| (1, 2048, 16,  64) | F      |  30.9 | 30.9 | **100%** ✓ |  48 MB |  520 MB |
| (1, 2048, 16,  64) | T      |  46.1 | 52.1 |        88% |  48 MB |  520 MB |
| (1, 4096, 16,  64) | F      |  35.9 | 35.5 | **101%** ✓ |  96 MB | 2064 MB |
| (1, 4096, 16,  64) | T      |  57.9 |  —   |          — |  96 MB | 2064 MB |
| (1, 1024,  8, 128) | F      |  10.3 | 21.4 |        48% |  16 MB |  132 MB |

At seq=4096 the FA backward uses **21.4× less peak memory** than eager. Throughput pattern matches the forward — at seq=1024 cuBLAS fp16 backward is itself tensor-core-fast and the FA tiling overhead doesn't pay back; the algorithmic win shows up at seq≥2048+causal.

**Honest call-out:** at seq=1024 we sit at 33-55% of the xformers Cutlass ceiling. cuBLAS fp16 matmul on V100 is already tensor-core-fast and Triton's kernel launch overhead dominates at short sequences. This matches the FA paper's own V100 numbers. To close the seq=1024 gap further would need persistent kernels (Volta doesn't support) or a single fused graph pass (Triton 2.3 limits). The full perf-steal breakdown and the optimisations adopted from xformers' autotune patterns (BSD-3 attribution) live in [`PERF.md`](PERF.md).

## Backward correctness

dQ/dK/dV max-abs error vs fp64 SDPA + autograd reference is < **3e-3** across all shapes (D ∈ {32, 64, 128}, causal/non-causal). Causal-leak test confirms dQ[i] is bit-identical when K[j>i] / V[j>i] are perturbed. Full breakdown in [`VERIFY.md`](VERIFY.md#backward).

## Stability

No NaN/Inf at seq up to 8192 (tested with causal and non-causal). The fully-masked-row guard (`m_curr_safe`, `safe_l`) handles the causal-row-0 edge case that otherwise NaNs out from `exp(-inf − (-inf))`.

## Real-model validation

Validated as a drop-in attention kernel on five HuggingFace targets on a V100 32GB. **One kernel, head_dim 64 + 128 paths, no per-model code:**

| model | family | head_dim | logits cos-sim | greedy match | prefill speedup @ seq=4096 |
|---|---|---:|---:|---:|---:|
| `gpt2` (124M) | MHA | 64 | 1.000000 | 50 / 50 | n/a (max seq 1024) |
| `Qwen/Qwen2.5-0.5B` | GQA 14:2 | 64 | 0.999998 | 50 / 50 | **1.85×** |
| `Qwen/Qwen2.5-7B` | GQA 28:4 | 128 | 1.000000 | 50 / 50 | **1.38×** |
| `Qwen/Qwen3-*` (1.7B/4B/8B) | GQA + QK-norm | 128 | patch wired up | — | requires transformers ≥ 4.51 |
| `state-spaces/mamba-130m-hf` | linear (SSM) | n/a | **refused** with clear `RuntimeError` | n/a | n/a |

On Qwen2.5-7B at seq=4096 the kernel removes **~4.3 GB** of per-attention-layer peak memory (eager 21 GB → kernel 16.7 GB). Eager fp16 attention on Qwen2.5-7B in fact produces NaN logits via QK^T overflow in late layers; the kernel's fp32 accumulator is required for correctness, not just speed.

### Real-model **backward** training (Qwen2.5-7B on a single V100 32 GB)

- Grad parity (cos-sim ≥ 0.998 on tracked weights, vs fp32 reference because eager fp16 backward NaNs on Qwen2.5-7B for the same QK^T overflow reason).
- **Max trainable sequence length: eager 1024 → FA 1792 (1.75×)**, purely from never materialising `(B, H, S, S)` during fwd+bwd.
- **4.4 GB saved** on full-model fwd+bwd peak at seq=1024 (28.8 → 24.3 GB) on Qwen2.5-7B.
- Single-line patch: every HF patch switched from `flash_attn_forward` → `flash_attn` to enable autograd through the kernel.

Patches live in `flash_attn_volta/patch_hf.py` (`patch_gpt2`, `patch_qwen2`, `patch_qwen3`, `patch_llama`, plus `patch_model` auto-dispatch that refuses Mamba/RWKV/RecurrentGemma/RetNet with a clear error). Full breakdown — what broke at each scale, fp32-reference parity test, per-seq throughput/memory, the QK-norm delta for Qwen3, the linear-attention safety check, the grad-parity matrix, the OOM-crossover table — in [`REAL_MODEL.md`](REAL_MODEL.md).

```python
from transformers import AutoModelForCausalLM
from flash_attn_volta.patch_hf import patch_qwen2, patch_model

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B",
                                              torch_dtype="float16",
                                              attn_implementation="eager").cuda()
patch_qwen2(model)            # routes attention through flash_attn_volta on prefill + training
# or: patch_model(model)       # auto-dispatch (raises on Mamba/RWKV/etc.)
```

## What's NOT here

- **fp8, bf16.** fp16 only. (V100 doesn't have native bf16 anyway.)
- **Variable-length / packed sequences.** Standard dense `(B, S, H, D)`.
- **Dropout, ALiBi, sliding window.** Forward+backward causal/non-causal is all there is.
- **MQA/GQA in the kernel.** Heads are 1:1 between Q and K/V at the kernel level; GQA models are handled in the HF patch via `repeat_kv` before calling the kernel (see `flash_attn_volta/patch_hf.py`).

## Layout

```
flash-attn-volta/
├── flash_attn_volta/
│   ├── triton_fa.py       # fwd + bwd Triton kernels (autotuned)
│   ├── patch_hf.py        # HF model patches (gpt2 / qwen2 / qwen3 / llama / auto-dispatch)
│   └── ref.py             # naive reference for testing
├── tests/                 # correctness, backward gradcheck, real-model parity
├── bench/                 # throughput + memory: synthetic, backward, real-model, max-seq
├── scripts/               # run_verify.sh, hf_push_dataset.py
├── probes/                # autotune sweep outputs
├── results/               # *.json from benchmark runs
└── TASK.md PLAN.md RESEARCH.md VERIFY.md RESULTS.md REAL_MODEL.md PERF.md
```

## Reproduce

```bash
git clone https://github.com/AlexWortega/flash-attn-volta && cd flash-attn-volta
pip install --user triton==2.3.0 torch
CUDA_VISIBLE_DEVICES=0 bash scripts/run_verify.sh    # correctness suite
CUDA_VISIBLE_DEVICES=0 python3 bench/bench.py        # forward TFLOP/s table
CUDA_VISIBLE_DEVICES=0 python3 bench/backward.py     # fwd+bwd TFLOP/s + memory
CUDA_VISIBLE_DEVICES=0 python3 bench/real_model.py   # real HF model bench
```

A V100 (or any SM 7.0 device) is required.

## License

Apache 2.0. Performance optimisations adopted in spirit (no source copy) from xformers' Triton autotune patterns; xformers is BSD-3.

## Credits

- FlashAttention: Tri Dao et al., [arXiv 2205.14135](https://arxiv.org/abs/2205.14135).
- xformers: Lefaudeux et al. (Meta), <https://github.com/facebookresearch/xformers>.
- Triton: Philippe Tillet et al.
- Built by `ml-intern` Claude Code skill: <https://github.com/AlexWortega/claude-ml-intern-skill>.
