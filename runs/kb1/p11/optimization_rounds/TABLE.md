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
| 1 | 1 | completed | True | True | 0.0287 | 540192 | 1.190 | 41.400 |
| 2 | 2 | completed | True | True | 0.0284 | 321368 | 1.180 | 41.500 |
| 3 | 3 | completed | True | True | 0.0260 | 819427 | 1.190 | 45.700 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p11/optimization_rounds/round1/output_model_new.py). It now uses a 4-wave `2x2` warp grid, stages operands through LDS, loads global data with `S.amdgpu.raw_buffer_load_x4`, issues `S.amdgpu.mfma_32x32x8_bf16_f32` twice per `K=16` chunk in natural half order, and writes results back with the fixed accumulator mapping. I validated it with the exact required round command. The latest result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p11/optimization_rounds/round1/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and `runtime_us: 42.0` on `gfx942`.
- total_tokens: 540192
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the pipelined MFMA kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p11/optimization_rounds/round2/output_model_new.py). The main change is a double-buffered LDS layout with split natural-half consumption of each 16-byte fragment, plus a steady-state K-loop unrolled by 2 tiles and an epilogue for the final pair. Validation passed with the exact required command. The round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p11/optimization_rounds/round2/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and `runtime_us: 42.0`.
- total_tokens: 321368
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-3 change in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p11/optimization_rounds/round3/output_model_new.py): the kernel now builds a range-backed descriptor for `C` and uses `S.amdgpu.raw_buffer_store_x1` in the epilogue instead of indexed tensor stores. Because `raw_buffer_store_x1` requires 32-bit data, the epilogue stages bf16 results in LDS, packs adjacent columns per even lane, and stores them as 32-bit words so OOB handling is descriptor-based rather than branch-based. Validation passed with the required command. The latest round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p11/optimization_rounds/round3/debug_eval.jsonl) reports `correctness: true` and `runtime_us: 46.5`.
- total_tokens: 819427
