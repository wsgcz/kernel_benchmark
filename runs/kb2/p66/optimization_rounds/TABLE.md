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
| 1 | 1 | completed | True | True | 0.0279 | 401836 | 0.248 | 8.880 |
| 2 | 2 | completed | True | True | 0.0287 | 269845 | 0.248 | 8.650 |
| 3 | 3 | completed | True | True | 0.0283 | 362490 | 0.247 | 8.720 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p66/optimization_rounds/round1/output_model_new.py). The new path uses `S.amdgpu.mfma_32x32x8_bf16_f32`, stages operand fragments through LDS, loads global data with `S.amdgpu.raw_buffer_load_x4`, scales from one wave to a `2 x 2` wave grid, and keeps softmax in Substrate so `forward()` does not use torch linear algebra. Validation passed with the exact required command. The round-local eval record in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p66/optimization_rounds/round1/debug_eval.jsonl) shows `compiled: true` and `correctness: true`.
- total_tokens: 401836
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 optimization in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p66/optimization_rounds/round2/output_model_new.py). The GEMM kernel now uses a 2-stage LDS double buffer, processes the K dimension in `32`-wide steady-state iterations (`2 x TILE_K`), and interleaves MFMA with split shared-memory refills so the next tile is staged while the current tile is being consumed. Validation passed with the required command. The round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p66/optimization_rounds/round2/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 8.65`.
- total_tokens: 269845
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-3 kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p66/optimization_rounds/round3/output_model_new.py). The GEMM path now uses ranged AMDGPU buffer resources for the global `X` and `W` `raw_buffer_load_x4` traffic, and the epilogue bias/mask fetches were moved to ranged raw-buffer loads as well while preserving the existing software-pipelined double-buffered MFMA structure. Validation passed with the required command. The round-local eval in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p66/optimization_rounds/round3/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 8.72`.
- total_tokens: 362490
