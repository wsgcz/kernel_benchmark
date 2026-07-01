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
| 1 | 1 | completed | True | True | 0.0193 | 371779 | 0.283 | 14.700 |
| 2 | 2 | completed | True | True | 0.0188 | 423750 | 0.280 | 14.900 |
| 3 | 3 | completed | True | True | 0.0189 | 375538 | 0.280 | 14.800 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p09/optimization_rounds/round1/output_model_new.py). It now uses a 4-wave `64x64` block tile, stages operands through LDS, loads global data with `S.amdgpu.raw_buffer_load_x4`, and issues two natural `S.amdgpu.mfma_32x32x8_bf16_f32` steps per `K=16` slice. I also kept the cached weight-transpose and bias path outside the kernel launch logic so the forward path remains cudagraph-safe. Validation passed with the exact required command. The latest round-local result in `debug_eval.jsonl` shows `compiled: true`, `correctness: true`, and `runtime_us: 14.7`.
- total_tokens: 371779
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p09/optimization_rounds/round2/output_model_new.py). The kernel keeps the required MFMA mapping, stages A/B through LDS, unrolls the K loop by 2 as `K=32` pairs, and uses ping-pong LDS buffers with a steady-state software pipeline that interleaves global loads, LDS writes, and four MFMA steps per pair. Validation passed with the exact required command. The round-local eval recorded `compiled: true`, `correctness: true`, and `runtime_us: 14.9` in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p09/optimization_rounds/round2/debug_eval.jsonl).
- total_tokens: 423750
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p09/optimization_rounds/round3/output_model_new.py). Validation passed with the required command. The round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p09/optimization_rounds/round3/debug_eval.jsonl) shows `compiled: true` and `correctness: true`.
- total_tokens: 375538
