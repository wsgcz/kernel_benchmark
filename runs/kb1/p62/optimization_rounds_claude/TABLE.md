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
| 1 | 1 | completed | True | True | - | 2664135 | - | - |
| 2 | 2 | completed | True | True | - | 274584 | - | - |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: claude
- summary: All tests passed. The implementation is complete. **Summary:** 1. **Created MFMA-optimized Conv2D kernel** in `output_model_new.py`: - Block tile: 128x128 - Warp grid: 2x2 (4 warps) - Per-warp tile: 64x64 (2x2 array of 32x32 MFMA tiles) - Uses `mfma_32x32x8_bf16_f32` instructions - Correct fragment layout: `lane_col = lane % 32`, `lane_k_base = (lane // 32) * 4` - K advances in MFMA-sized chunks: `k = k_tile * 8 + lane_k_base + e` - Fixed accumulator writeback layout per prompt specifications 2. **Updated test file** (`test_conv2d_asym.py`) to import from p62 and test the correct kernel shape (in_channels=32, out_channels=64, kernel_size=(5,9)) 3. **Updated benchmark file** (`bench_conv2d.py`) with the correct import path and kernel configuration 4. **Created `case.txt`** with the round directory path 5. **All 3 validation tests passed**: - `test_mfma_conv2d_benchmark_shape` - correctness validation - `test_mfma_conv2d_output_shape` - output shape verification - `test_mfma_conv2d_rejects_wrong_shape` - input validation
- total_tokens: 2664135
### round2
- status: completed
- unlocked hints: 2
- agent: claude
- summary: All 3 tests passed. The Split-K optimization has been successfully implemented with: 1. **Split-K decomposition**: `SPLIT_K_SLICES = 2` with channels partitioned contiguously and disjointly 2. **Extended launch grid**: Grid extended in the `x` dimension by `SPLIT_K_SLICES` 3. **Channel partition invariant**: `c_per_split = ceil_div(in_channels, SPLIT_K_SLICES)` with `c_start` and `c_end` properly computed 4. **K linearization**: Changed to `k_idx = c * kernel_area + kh * kW + kw` for channel-aligned split boundaries 5. **Operand layout**: Direct NCHW / OIHW tensor reads without relying on NHWC / OHWI pre-permutation 6. **FP32 workspace reduction**: Uses `S.amdgpu.buffer_atomic_add_f32` to reduce partial sums into a GEMM-major fp32 workspace 7. **Finalization kernel**: Separate kernel that converts fp32 workspace back to bf16 NCHW output 8. **Cudagraph safety**: Workspace tensor is pre-allocated and reused, only zeroed between runs
- total_tokens: 274584
