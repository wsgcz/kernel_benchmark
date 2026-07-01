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
| 1 | 1 | completed | True | True | 0.7216 | 515413 | 0.210 | 0.291 |
| 2 | 2 | completed | True | True | 1.6308 | 127921 | 0.212 | 0.130 |
| 3 | 3 | completed | True | True | 1.6031 | 465085 | 0.210 | 0.131 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p12/optimization_rounds_no_invariants/round1/output_model_new.py). It now launches a `64x64` tile over a `2 x 2` wave grid (`256` threads), uses `S.amdgpu.raw_buffer_load_x4` to stage operands through LDS, and issues `S.amdgpu.mfma_32x32x8_bf16_f32` twice per LDS fragment in natural half order. Validation passed with the required harness command. The latest round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p12/optimization_rounds_no_invariants/round1/debug_eval.jsonl) shows `correctness: true` and candidate runtime `0.29 us` versus reference `0.61 us`.
- total_tokens: 515413
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p12/optimization_rounds_no_invariants/round2/output_model_new.py). The updated kernel keeps the 4-wave `2 x 2` warp grid, stages `A` and `B` through double-buffered LDS with `raw_buffer_load_x4`, interleaves the two natural BF16 MFMA halves with the next buffer fill, and writes the final diagonal-left result from staged data instead of rereading `B` from global memory. Validation passed with the required command. The harness result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p12/optimization_rounds_no_invariants/round2/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and candidate runtime `0.13 us` vs reference `0.211 us`.
- total_tokens: 127921
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round output kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p12/optimization_rounds_no_invariants/round3/output_model_new.py) without modifying `candidate_input.py`. Validation passed with the required command. Round-local debug eval reported `correctness: true` and `runtime_us: 0.131` versus `ref_runtime_us: 0.214`.
- total_tokens: 465085
