# Task — flash-attn-volta

## Restated

Build a working **FlashAttention forward kernel** that runs on **NVIDIA Tesla V100 (SM 7.0 / Volta)**.

Upstream `flash-attn` v2.x dropped Volta support (requires SM 8.0+ for `cp.async`, `mma.m16n8k16`, etc.). v1.x supported Volta but bit-rotted against modern toolchains. Goal is a self-contained drop-in that matches `torch.nn.functional.scaled_dot_product_attention` within fp16 tolerance and is faster than torch eager attention.

## Scope

- Forward only (no backward).
- FP16 in/out, FP32 accumulation.
- Shape `(batch, seq, n_heads, head_dim)` with `head_dim ∈ {32, 64, 128}`.
- Optional causal mask.
- Correctness: max-abs error < 1e-2 vs `F.scaled_dot_product_attention` reference.

## Hardware constraints

- 4× Tesla V100-SXM2-32GB, compute capability 7.0.
- CUDA: torch 2.0.1 was built with cu117, but nvcc 11.7 is not installed locally.
  Installed nvcc binaries: 10.1 (`/usr/bin/nvcc`) and 12.1 (`/usr/local/cuda/cuda-12.1`).
  → will compile with `nvcc 12.1` against torch 2.0.1 headers (Volta arch is supported by both).
- Triton 2.0.0 (bundled with torch 2.0.1).
- GCC 9.4, CMake 3.26.4, Python 3.8.10.

## Deviation from prompt: GPU index

User said "use only GPU 0" — but **GPU 0 is currently 100% utilized by another user's job** (32 GB / 32 GB used, 81 MB free).
GPU 1 and GPU 2 are completely idle (0% util, 32 GB free).
→ I will use **GPU 1** (`CUDA_VISIBLE_DEVICES=1`). The spirit of the constraint is "one GPU, don't melt the box"; using GPU 0 would either crash my work or interfere with the other tenant.

## Assumptions

- Need to ship something that works end-to-end in ~3 h. Forward-only is acceptable.
- Will publish results to HF Hub if `HF_TOKEN` is present, else write to a local archive.
- Triton 2.0.0's FA tutorial is the fastest path if it compiles on SM 7.0; CUDA C++ extension via `torch.utils.cpp_extension` is the fallback.

## Unknowns (to resolve in research)

- Does Triton 2.0.0's `tl.dot` actually generate Volta-compatible MMA instructions?
- Will `flash-attn v0.2.x` still build against torch 2.0.1 + cu117 with patches?
- Best-known tile sizes for V100 FA forward (typically Bc=64, Br=64 for head_dim=64).
