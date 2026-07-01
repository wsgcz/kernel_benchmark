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
| 1 | 1 | completed | True | True | 0.0355 | 760236 | 0.267 | 7.520 |
| 2 | 2 | completed | True | True | 0.0360 | 213364 | 0.274 | 7.620 |
| 3 | 3 | completed | True | True | 0.0356 | 143320 | 0.266 | 7.470 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p86/optimization_rounds/round1/output_model_new.py). It now uses `S.amdgpu.mfma_32x32x8_bf16_f32`, stages operands through LDS, uses `S.amdgpu.raw_buffer_load_x4` for global operand loads, scales to 4 waves as a `2 x 2` warp grid, and keeps cached weight/bias tensors for cudagraph-safe reuse. Validation passed with the required command. The latest round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p86/optimization_rounds/round1/debug_eval.jsonl) shows `correctness: true` and `runtime_us: 7.46`.
- total_tokens: 760236
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 optimization in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p86/optimization_rounds/round2/output_model_new.py). The kernel now uses a double-buffered LDS layout for A/B, unrolls the K-loop by 2, and issues the next pair of `raw_buffer_load_x4` loads between MFMA steps while preserving the existing 4-wave 2x2 warp-grid MFMA mapping and the required accumulator writeback layout. Validation passed with the exact required command. The round-local eval reported `compiled: true`, `correctness: true`, and candidate `runtime_us: 7.38`.
- total_tokens: 213364
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-3 change in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p86/optimization_rounds/round3/output_model_new.py). The K-loop tail prefetch branches were removed so the pipelined `raw_buffer_load_x4` path now relies on the resource `range` semantics for safe OOB-zero behavior on the final iteration, while preserving the existing double-buffered MFMA/LDS overlap. Validation passed with the required harness command. The recorded result in `debug_eval.jsonl` shows `compiled: true` and `correctness: true`.
- total_tokens: 143320
