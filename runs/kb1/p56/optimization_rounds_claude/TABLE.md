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
| 1 | 1 | completed | True | True | - | 16353259 | - | - |
| 2 | 2 | completed | True | True | - | 1731598 | - | - |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: claude
- summary: Both tests pass. The p56 MFMA Conv2D kernel is now working correctly with: 1. **Writeback fix**: Corrected the MFMA accumulator layout formula from `row = tile_row_base + 16 * pack + acc_idx` to `row = tile_row_base + 8 * (acc_idx // 4) + 4 * pack + (acc_idx % 4)` 2. **Output layout fix**: Changed the output tensor from GEMM-major to NCHW layout 3. **Weight caching fix**: Always convert weight to bf16 to handle in-place weight modifications The kernel achieves max absolute difference of 0.00778, well within bf16 precision tolerance.
- total_tokens: 16353259
### round2
- status: completed
- unlocked hints: 2
- agent: claude
- summary: Both tests pass. The Split-K optimization has been successfully implemented with: 1. `SPLIT_K_SLICES = 2` - Extended the grid by 2x 2. Block decomposition: `linear_block_id = tile_block_id * SPLIT_K_SLICES + split_k_id` 3. Channel partition: Each split handles `c_per_split = 32` channels 4. K linearization: `k_idx = c * kernel_area + kh * KERNEL_W + kw` (channel-major) 5. Accumulator/output: Uses fp32 workspace with GEMM-major indexing 6. Reduction: `S.amdgpu.buffer_atomic_add_f32` for atomic accumulation 7. Finalization: Separate kernel converts fp32 workspace back to bf16 NCHW output The kernel preserves the existing MFMA lane mapping and accumulator-to-output mapping as required.
- total_tokens: 1731598
