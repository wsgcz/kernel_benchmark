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
| 1 | 1 | completed | True | True | 0.0081 | 996162 | 0.538 | 66.100 |
| 2 | 2 | completed | True | True | 0.0175 | 581711 | 0.548 | 31.300 |
| 3 | 3 | completed | True | True | 0.0177 | 305573 | 0.552 | 31.100 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p03/optimization_rounds/round1/output_model_new.py) and left `candidate_input.py` unchanged. The kernel now uses `S.amdgpu.mfma_32x32x8_bf16_f32`, `S.amdgpu.raw_buffer_load_x4`, a 256-thread block as a 2x2 wave grid, LDS staging, and the fixed accumulator writeback mapping. I also made the forward path graph-safe by avoiding any descriptor/device-metadata construction in `forward()`. Validation with the exact provided command passed correctness. The latest debug eval result reports `correctness: true` with candidate runtime about `90.1 us` on `gfx942`, recorded in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p03/optimization_rounds/round1/debug_eval.jsonl).
- total_tokens: 996162
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p03/optimization_rounds/round2/output_model_new.py). The updated kernel keeps the four-wave 2x2 warp tiling, stages operands through LDS as per-lane `u32 x4` fragments, reinterprets them into two natural BF16 MFMA halves, and uses a prologue / steady-state / epilogue software pipeline with double buffering and K-loop unrolling by 2. Validation passed with the required harness command. The round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p03/optimization_rounds/round2/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 31.0`.
- total_tokens: 581711
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-local kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p03/optimization_rounds/round3/output_model_new.py). The kernel keeps the existing software-pipelined, double-buffered MFMA structure and uses ranged AMDGPU raw-buffer loads for the global A/B fetch path. Validation passed with the required command. The round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p03/optimization_rounds/round3/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 31.3`.
- total_tokens: 305573
