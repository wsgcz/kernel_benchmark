# GEMM Optimization History

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
| 1 | 1 | completed | True | True | 0.0720 | 671705 | 0.522 | 7.250 |
| 2 | 2 | completed | True | True | 0.0315 | 244966 | 0.229 | 7.280 |
| 3 | 3 | completed | True | True | 0.0320 | 222638 | 0.233 | 7.290 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p01/optimization_rounds_no_invariants/round1/output_model_new.py). It now uses `S.amdgpu.mfma_32x32x8_bf16_f32`, stages operands through LDS, uses `S.amdgpu.raw_buffer_load_x4` for global loads, and scales the MFMA path to a 2x2 warp grid with per-warp fetch/writeback offsets. Validation passed with the required round-local command. The harness result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p01/optimization_rounds_no_invariants/round1/debug_eval.jsonl) reports `compiled: true` and `correctness: true`.
- total_tokens: 671705
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p01/optimization_rounds_no_invariants/round2/output_model_new.py). The kernel now keeps the 4-wave `2 x 2` warp grid and MFMA mapping intact, stages A/B through double-buffered LDS, uses `raw_buffer_load_x4` 16-byte loads, and processes K in unrolled pairs so global loads for the second 16-wide slice overlap with MFMA on the first slice. Validation passed with the exact required command. The round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p01/optimization_rounds_no_invariants/round2/debug_eval.jsonl) shows `correctness: true` with measured candidate runtime `7.25 us`.
- total_tokens: 244966
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-3 change in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p01/optimization_rounds_no_invariants/round3/output_model_new.py): the K-loop now unconditionally prefetches the next double-buffered tile, relying on the existing ranged `S.amdgpu.raw_buffer_load_x4` resources to zero-fill the final OOB prefetch instead of branching on `ko_pair + 1 != K_TILE_PAIRS`. That removes the explicit end-of-loop OOB guard without changing the pipelined MFMA/LDS structure. Validation passed with the required command. The round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p01/optimization_rounds_no_invariants/round3/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and `runtime_us: 7.37`.
- total_tokens: 222638
