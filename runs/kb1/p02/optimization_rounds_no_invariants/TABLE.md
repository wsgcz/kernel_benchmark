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
| 1 | 1 | completed | True | True | 0.0300 | 1546896 | 0.236 | 7.860 |
| 2 | 2 | completed | True | True | 0.0294 | 151464 | 0.234 | 7.970 |
| 3 | 3 | completed | True | True | 0.0312 | 358409 | 0.236 | 7.560 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p02/optimization_rounds_no_invariants/round1/output_model_new.py). It uses `S.amdgpu.mfma_32x32x8_bf16_f32`, vectorized `raw_buffer_load_x4`, LDS staging, and a 4-wave `2 x 2` warp-grid tile. Validation passed with the required harness command. The round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p02/optimization_rounds_no_invariants/round1/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and candidate `runtime_us: 7.99`.
- total_tokens: 1546896
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the software-pipelined MFMA kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p02/optimization_rounds_no_invariants/round2/output_model_new.py). The kernel now uses double-buffered LDS staging for A/B, unrolls the K loop by 2 tiles, and overlaps global `raw_buffer_load_x4` prefetches with MFMA consumption by reloading the just-consumed buffer while the other buffered tile is executing. Validation passed with the required round-local command. The debug eval in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p02/optimization_rounds_no_invariants/round2/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and candidate `runtime_us: 7.96`.
- total_tokens: 151464
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round change in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p02/optimization_rounds_no_invariants/round3/output_model_new.py): the hot K-pair loop is now split into a steady-state loop plus a final epilogue pair, which removes the two per-iteration tail branches from the pipelined section while preserving the existing double-buffered MFMA/LDS/global-load structure. The kernel continues to use ranged AMDGPU buffer resources for the raw loads. Validation passed with the required command. The round-local eval in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p02/optimization_rounds_no_invariants/round3/debug_eval.jsonl) reports `compiled: true` and `correctness: true`.
- total_tokens: 358409
