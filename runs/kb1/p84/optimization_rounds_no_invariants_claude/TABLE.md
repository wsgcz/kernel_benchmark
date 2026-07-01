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
| 1 | 1 | agent_failed | None | None | - | 13252617 | - | - |
| 2 | 2 | completed | True | True | - | 8928502 | - | - |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: agent_failed
- unlocked hints: 1
- agent: claude
- summary: Agent produced no textual summary.
- total_tokens: 13252617
- error: Agent exited with code -1
### round2
- status: completed
- unlocked hints: 2
- agent: claude
- summary: All tests pass. The kernel is now working correctly. Here's a summary of the fixes: **Issues Fixed:** 1. **MFMA Row Permutation**: Added `_mfma_permute_row` function to handle the MFMA 32x32x8 lane-to-row mapping. Lane `l` must load row `permute_row(l % 32)` within each MFMA block. 2. **Warp Base Calculation**: Changed from `warp_row * MFMA_M` to `warp_row * WARP_TILE_M` to correctly partition rows among warps (warp 0 handles rows 0-63, warp 2 handles rows 64-127). 3. **K Iteration**: Added inner loop `for k_iter in S.range(K_BLOCK_SLICES)` to process all K values within each k_block. Each MFMA processes K=8, but GROUP_K=32 requires 4 MFMA calls. 4. **Writeback Coordinates**: Fixed to include warp base offsets (`row_base`, `col_base`) when computing output positions. The kernel now correctly computes Conv2D with Split-K reduction, achieving results within bf16 tolerance (max diff ~0.013, well within atol=0.1).
- total_tokens: 8928502
