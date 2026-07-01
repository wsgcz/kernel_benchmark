# Conv2D Optimization History

This file is shared state between optimization rounds.
The orchestrator rewrites the history section after each round while preserving this intro.

Interpretation rules for the agent:
- Treat the recorded outcomes as the authoritative history of what has already been tried.
- Avoid repeating a failed optimization unless the new round has a clearly different reason to retry it.
- Prefer incremental decisions that build on the previous round instead of restarting from scratch.

<!-- AUTO-GENERATED HISTORY BELOW -->

| round | prompts | status | compiled | correctness | speedup | total_tokens | ref_us | new_us |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | seed baseline | seeded | None | None | - | - | - | - |
| 1 | 1 | completed | True | True | - | 10571323 | - | - |
| 2 | 2 | completed | True | True | - | 499208 | - | - |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: claude
- summary: Both tests pass. The MFMA-optimized depthwise Conv2D kernel is complete and working correctly. **Summary of the solution:** - Uses `mfma_32x32x8_bf16_f32` instructions for AMD MI300 GPUs - Block tile: one block per (batch, channel, row_group), processes 32 output rows - MFMA dimensions: M=32 (rows), N=32 (columns), K=8 (kernel elements) - Key insight: A fragment = weights (constant, broadcasts across columns), B fragment = input values (varies by output column) - No LDS staging, vectorized loads, or async copies as required The kernel correctly implements depthwise convolution with the specified asymmetric kernel size (3x7) for input shape (32, 128, 128, 256).
- total_tokens: 10571323
### round2
- status: completed
- unlocked hints: 2
- agent: claude
- summary: The validation tests passed successfully. The Split-K implementation is working correctly. Let me verify the implementation meets all requirements: 1. **Block decomposition invariant**: `linear_block_id = tile_block_id * SPLIT_K_SLICES + split_k_id` ✅ 2. **K partition for depthwise conv**: Kernel elements split across 2 slices ✅ 3. **FP32 accumulation with atomic add**: Uses `S.amdgpu.buffer_atomic_add_f32` ✅ 4. **Separate finalize kernel**: Converts fp32 workspace to bf16 NCHW output ✅ 5. **MFMA instructions used**: `S.amdgpu.mfma_32x32x8_bf16_f32` ✅ 6. **Cudagraph-safe**: Workspace pre-allocated and reused based on storage pointer ✅ The implementation: - Extends launch grid by `SPLIT_K_SLICES = 2` - Each split processes half of the kernel elements (11 elements each) - Partial sums accumulated in fp32 workspace via atomic adds - Finalize kernel converts workspace to final bf16 NCHW output - No torch native compute used for convolution Tests passed in 4.68 seconds with correct results.
- total_tokens: 499208
