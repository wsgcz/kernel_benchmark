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
| 1 | 1 | completed | True | True | - | 3293510 | - | - |
| 2 | 2 | completed | True | True | - | 4522543 | - | - |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: claude
- summary: All 3 tests pass. The MFMA-optimized Conv2D kernel is complete and validated: - **test_mfma_conv2d_benchmark_shape**: PASSED - Numerical correctness (max diff ~0.008 within bf16 tolerance) - **test_mfma_conv2d_output_shape**: PASSED - Output shape (8, 64, 508, 504) - **test_mfma_conv2d_rejects_wrong_shape**: PASSED - Input validation works The kernel successfully transforms the naive f32 element-wise convolution to an MFMA-based implicit GEMM implementation using 32x32x8 bf16→f32 matrix instructions.
- total_tokens: 3293510
### round2
- status: completed
- unlocked hints: 2
- agent: claude
- summary: The implementation is complete. Here's a summary: **Implementation Summary:** - Created MFMA-optimized Conv2D kernel with Split-K reduction for asymmetric kernel (5x9) - Split-K divides input channels into 2 slices (channels 0-15 and 16-31) - Each slice is computed by a separate kernel launch writing to its own workspace - A reduction kernel combines the partial sums - Store kernel converts fp32 workspace to bf16 NCHW output **Key Design Decisions:** - Used sequential kernel launches for Split-K slices instead of parallel blocks with atomic reduction, as `buffer_atomic_add_f32` had issues with large buffer sizes - Followed the candidate_input.py structure for thread-to-output mapping (lane-based distribution, 4 rows per lane, MFMA_K=8 chunks) - Implemented K linearization: `k_idx = c * kernel_area + kh * kernel_w + kw` **Tests Passed:** - `test_mfma_conv2d_benchmark_shape` - correctness validation - `test_mfma_conv2d_output_shape` - output shape (8, 64, 508, 504) - `test_mfma_conv2d_rejects_wrong_shape` - input validation
- total_tokens: 4522543
