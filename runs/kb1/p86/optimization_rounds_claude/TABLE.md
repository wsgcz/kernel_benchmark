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
| 1 | 1 | completed | True | True | - | 2717036 | - | - |
| 2 | 2 | completed | True | True | - | 513885 | - | - |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: claude
- summary: The implementation is complete. Here's a summary of what was done: 1. **Read the starting kernel** from `candidate_input.py` - a naive fused depthwise + pointwise Conv2D kernel 2. **Created the MFMA-optimized kernel** in `output_model_new.py`: - Block tile: 128 × 128 - 4 warps in 2×2 grid - Each warp computes 64×64 tile (2×2 array of 32×32 MFMA tiles) - Uses `mfma_32x32x8_bf16_f32` instructions with BF16 inputs and F32 accumulators - Follows the specified fragment loading pattern: - `lane_col = lane % 32` - `lane_k_base = (lane // 32) * 4` - `k = k_tile * 8 + lane_k_base + e` - Follows the specified writeback pattern for MFMA accumulator unpacking 3. **Validation results:** - All 3 tests pass - Benchmark runs successfully: ~16ms per iteration, 4.5 TFLOPS - Max absolute difference: 0.009 (well within bf16 tolerance) 4. **Updated callsites:** - `/workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py` - `/workspace/substrate/benchmark/conv2d/bench_conv2d.py` 5. **Created `case.txt`** with the absolute path of the round directory
- total_tokens: 2717036
### round2
- status: completed
- unlocked hints: 2
- agent: claude
- summary: All 3 validation tests passed. The Split-K implementation with `SPLIT_K_SLICES = 2` is complete: **Key changes implemented:** 1. **Block decomposition**: `linear_block_id = tile_block_id * SPLIT_K_SLICES + split_k_id` - extends the grid in the x dimension while preserving M/N tile ownership 2. **Channel partition**: Each split handles `c_per_split = ceil_div(in_channels, SPLIT_K_SLICES)` contiguous channels. Split 0: channels [0, 32), Split 1: channels [32, 64) 3. **K linearization**: `k_idx = c * kernel_area + kh * kernel_w + kw` (channel-major instead of spatial-major), ensuring split boundaries are channel-aligned 4. **Operand layout**: Reads input and weights directly as NCHW/OIHW tensors without pre-permutation 5. **Reduction**: All splits accumulate into an fp32 workspace using `S.amdgpu.buffer_atomic_add_f32` with GEMM-major indexing `linear_idx = row * gemm_n + col` 6. **Finalization**: A separate `store_kernel` converts the fp32 workspace back to bf16 NCHW output layout after all split reductions complete
- total_tokens: 513885
