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
| 1 | 1 | completed | True | True | 0.2006 | 3152052 | 0.728 | 3.630 |
| 2 | 2 | completed | True | True | 0.1918 | 387340 | 0.748 | 3.900 |
| 3 | 3 | completed | True | True | 0.1001 | 419927 | 0.737 | 7.360 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the standalone MFMA kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p09/optimization_rounds_no_invariants/round1/output_model_new.py). The final path uses `S.amdgpu.mfma_32x32x8_bf16_f32`, vectorized global `raw_buffer_load_x4`, stages A and B through LDS, and scales the per-wave MFMA invariant across a 2x2 wave grid for a 64x64 block tile. The torch matmul fallback is removed. Validation passed with the required harness command. The recorded result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p09/optimization_rounds_no_invariants/round1/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and candidate `runtime_us: 3.63`.
- total_tokens: 3152052
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p09/optimization_rounds_no_invariants/round2/output_model_new.py). The kernel keeps the 4-wave `2 x 2` warp grid and MFMA mapping, stages A/B through double-buffered LDS, and pipelines split 16-byte operand fragment loads from LDS against the two natural `mfma_32x32x8_bf16_f32` steps. Validation passed with the exact required harness command. The recorded result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p09/optimization_rounds_no_invariants/round2/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and `runtime_us: 3.9`.
- total_tokens: 387340
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round’s change in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p09/optimization_rounds_no_invariants/round3/output_model_new.py): MFMA results now spill into a shared output tile and are written out with `S.amdgpu.raw_buffer_store_x4` through a `range`-backed per-row subview resource, so the output path also relies on hardware OOB discard instead of explicit guards. Validation passed with the required command. The latest round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p09/optimization_rounds_no_invariants/round3/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and `runtime_us: 7.36`.
- total_tokens: 419927
