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
| 1 | 1 | completed | True | True | 0.0218 | 676364 | 0.218 | 10.000 |
| 2 | 2 | completed | True | True | 0.0232 | 993463 | 0.217 | 9.340 |
| 3 | 3 | completed | True | True | 0.0233 | 304641 | 0.217 | 9.310 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p14/optimization_rounds/round1/output_model_new.py). It now uses a 4-wave `64x64x16` tiled MFMA path with `S.amdgpu.mfma_32x32x8_bf16_f32`, stages operands through LDS, and uses `S.amdgpu.raw_buffer_load_x4` for global operand fetches. The torch matmul fallback was removed; unsupported shapes/dtypes now raise. Validation passed with the required command. The round-local eval record reports `compiled: true` and `correctness: true` in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p14/optimization_rounds/round1/debug_eval.jsonl). Reported candidate runtime was `10.2 us` on `gfx942`.
- total_tokens: 676364
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p14/optimization_rounds/round2/output_model_new.py). The kernel keeps the MFMA path and MFMA/writeback invariants intact, adds two-stage LDS buffers, and unrolls the K loop by 2 so each iteration consumes two `mfma_32x32x8_bf16_f32` steps before refilling the staged operands. Validation passed with the required round-local command. The latest `debug_eval.jsonl` shows `correctness: true` and candidate runtime `9.34 us` with `cudagraph` timing.
- total_tokens: 993463
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p14/optimization_rounds/round3/output_model_new.py). The change keeps the double-buffered MFMA pipeline, removes the branchy steady-state K-prefetch path by using a fixed branchless main loop plus a final epilogue pair, and keeps ranged raw buffer resources for the global loads. I validated it with the required harness command; the result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p14/optimization_rounds/round3/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and `runtime_us: 9.32`.
- total_tokens: 304641
