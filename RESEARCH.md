# Research — flash-attn-volta

## FlashAttention algorithm

- Paper: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", arXiv 2205.14135.
- Core idea: **tile** the (Q, K, V) matmul so each tile fits in SRAM; compute softmax **incrementally** using the running-max / running-sum trick (online softmax) so we never materialize the full N×N attention matrix in HBM. Memory goes from O(N²) → O(N).

## Online softmax recap (per query tile)

```
m_i, l_i, O_i  =  -inf, 0, 0
for K_j, V_j in K, V:                       # streamed K/V tiles
    S_ij        = Q_i @ K_j.T * scale
    if causal:  S_ij = mask(S_ij)
    m_new       = max(m_i, max(S_ij, dim=-1))
    P_ij        = exp(S_ij - m_new)
    alpha       = exp(m_i - m_new)
    l_i         = alpha * l_i + sum(P_ij, dim=-1)
    O_i         = alpha * O_i + P_ij @ V_j   # in fp32
    m_i         = m_new
O_i  /=  l_i
```

## Approach decision

Three options were on the table:

1. **Port FA1 v0.2.x.** Repo `Dao-AILab/flash-attention@v0.2.8` was the last Volta-supporting tag. Heavily depends on a pinned CUTLASS, custom block-sparse path, and pre-cu118 `at::cuda` helpers. Patching against torch 2.0.1 + cu117 is doable but eats most of the budget; lots of moving parts.
2. **Hand-rolled CUDA C++ extension** using PTX `mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32` (the Volta wmma instruction). Maximum control, but writing + debugging a correct tiled FA kernel in raw CUDA is ~6+ h of careful work.
3. **Triton 2.0.0 kernel.** Triton's `tl.dot` lowers to the right Volta MMA on SM 7.0 (verified by probe — `probes/probe_triton.py` ran a 256×256 fp16 matmul on V100 with 0.03 max-abs error). The official Triton FA tutorial (`openai/triton/python/tutorials/06-fused-attention.py @ v2.0.0`) was written for Volta originally.

**Picked option 3.** Reasons:
- Probe confirms Triton emits working MMA on V100.
- Most code reuse → fastest path to a verifiable, benchmarkable kernel.
- One-file, no build system, no toolchain mismatch.
- Falls back gracefully: if Triton dies on some head_dim, we still own the algorithm.

## Reference implementations consulted

- Triton FA tutorial v2.0.0: github.com/openai/triton/blob/v2.0.0/python/tutorials/06-fused-attention.py
- FA1 v0.2.x kernels: github.com/Dao-AILab/flash-attention/tree/v0.2.8/csrc/flash_attn/src
- Online softmax trick: Milakov & Gimelshein 2018, arXiv 1805.02867

## Known V100 / Triton 2.0 pitfalls

- Triton 2.0 occasionally miscompiles when block sizes don't divide `head_dim`. Stick to `BLOCK_M=128`, `BLOCK_N=64` (or 32) for `head_dim ∈ {32, 64, 128}`.
- Triton's `tl.where(mask, value, -inf)` works but `tl.where(mask, value, float("-inf"))` needs a `.to(tl.float32)` cast on Volta. Use a large negative constant (`-1e9`) for masking instead.
- Volta has 96 KB SRAM per SM; tile sizes need to fit `(BLOCK_M*head_dim + 2*BLOCK_N*head_dim) * 2 bytes + acc/m/l` comfortably. For BLOCK_M=128, BLOCK_N=64, head_dim=128: 128*128*2 + 2*64*128*2 = 32K + 32K = 64K — within budget.
- `tl.exp` on Volta is fine; `tl.exp2` works too.
