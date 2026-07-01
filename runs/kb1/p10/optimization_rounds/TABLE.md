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
| 1 | 1 | completed | True | True | 0.0296 | 1424679 | 0.091 | 3.080 |
| 2 | 2 | completed | True | True | 0.0264 | 258706 | 0.090 | 3.410 |
| 3 | 3 | completed | True | True | 0.0308 | 435608 | 0.092 | 2.980 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p10/optimization_rounds/round1/output_model_new.py). It uses `S.amdgpu.mfma_32x32x8_bf16_f32`, stages operands through LDS, uses `S.amdgpu.raw_buffer_load_x4` for global loads, and scales the tile across a 2x2 wave grid. Validation passed with the required command. The latest round-local result in `debug_eval.jsonl` shows `correctness: true` and candidate runtime `3.09 us`.
- total_tokens: 1424679
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the pipelined MFMA kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p10/optimization_rounds/round2/output_model_new.py). It keeps the 4-wave 2x2 warp-grid mapping and the required MFMA/writeback invariants, adds LDS double buffering, and unrolls the K loop by 2 so the next tile’s global loads are issued between the two MFMA steps of the current tile. Validation passed with the exact round-local command. `debug_eval.jsonl` reports `compiled: true`, `correctness: true`, and `runtime_us: 3.04`.
- total_tokens: 258706
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Wrote the final kernel to [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p10/optimization_rounds/round3/output_model_new.py) and left [candidate_input.py](/workspace/kb_eval_pipeline/runs/kb1/p10/optimization_rounds/round3/candidate_input.py) unchanged. Validation passed with the required command. The round-local eval log in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p10/optimization_rounds/round3/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 2.98`.
- total_tokens: 435608
