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
| 1 | 1 | completed | True | True | 37.9834 | 284713 | 0.275 | 0.007 |
| 2 | 2 | completed | True | True | 0.1438 | 373998 | 0.279 | 1.940 |
| 3 | 3 | completed | True | True | 0.1171 | 190262 | 0.281 | 2.400 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p80/optimization_rounds_no_invariants/round1/output_model_new.py). It now uses a standalone Substrate AMDGPU path with `raw_buffer_load_x4`, LDS staging via `S.make_shared`, and two natural-order `S.amdgpu.mfma_32x32x8_bf16_f32` steps per warp fragment, scaled to 4 waves as a `2 x 2` warp grid. Validation passed with the required round-local command. The harness reported `compiled: true` and `correctness: true` in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p80/optimization_rounds_no_invariants/round1/debug_eval.jsonl).
- total_tokens: 284713
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p80/optimization_rounds_no_invariants/round2/output_model_new.py). The kernel now uses `S.amdgpu.mfma_32x32x8_bf16_f32`, stages operands through LDS with `raw_buffer_load_x4`, uses a 4-wave `2 x 2` warp grid, and applies a double-buffered, K-unrolled-by-2 software pipeline structure. Validation passed with the exact required command. The round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p80/optimization_rounds_no_invariants/round2/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 1.94`.
- total_tokens: 373998
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-3 optimization in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p80/optimization_rounds_no_invariants/round3/output_model_new.py). The kernel now creates row-scoped AMDGPU raw-buffer resources with `range_bytes` and removes the explicit K-tail guard branches, relying on `raw_buffer_load_x4(..., range=...)` to zero-fill OOB accesses safely while keeping the existing double-buffered pipelined structure. Validation passed with the required command. The round-local debug result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p80/optimization_rounds_no_invariants/round3/debug_eval.jsonl) shows `compiled: true` and `correctness: true`.
- total_tokens: 190262
