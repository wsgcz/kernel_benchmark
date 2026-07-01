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
| 1 | 1 | completed | True | True | 0.0349 | 452516 | 0.261 | 7.480 |
| 2 | 2 | completed | True | True | 0.0350 | 667315 | 0.260 | 7.430 |
| 3 | 3 | completed | True | True | 0.0351 | 254229 | 0.261 | 7.430 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p12/optimization_rounds/round1/output_model_new.py). It now uses a 4-wave `64x64x16` tiled MFMA path with `S.amdgpu.mfma_32x32x8_bf16_f32`, stages operands through LDS, loads global data with `S.amdgpu.raw_buffer_load_x4`, applies the fixed accumulator writeback mapping, and keeps the cached weight/bias path cudagraph-safe without using torch matmul-style compute. Validation passed with the required command: correctness `true` (`1 / 1`), candidate runtime `7.47 us`.
- total_tokens: 452516
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p12/optimization_rounds/round2/output_model_new.py) only. The kernel keeps the 4-wave MFMA mapping and fixed accumulator unpacking, and restructures the K path into a 2-step steady-state pipeline with ping-pong LDS stages. Global `raw_buffer_load_x4` loads are interleaved with MFMA and LDS staging so the next/next-next `K=16` tiles are prepared while the current tile is being consumed. Validation passed with the required harness command. The round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p12/optimization_rounds/round2/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and candidate `runtime_us: 7.43`.
- total_tokens: 667315
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p12/optimization_rounds/round3/output_model_new.py) only. I left `candidate_input.py` unchanged. The update removes the K-loop control branches by peeling the loop around the double-buffered pipeline, keeping the MFMA/LDS/global-load overlap structure and the unroll-by-2 schedule intact. Validation passed with `correctness: true` on the required harness command, and the recorded candidate runtime was `7.8 us` in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p12/optimization_rounds/round3/debug_eval.jsonl).
- total_tokens: 254229
