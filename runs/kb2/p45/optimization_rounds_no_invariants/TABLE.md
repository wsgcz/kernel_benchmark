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
| 1 | 1 | completed | True | True | 0.0134 | 520839 | 0.899 | 66.900 |
| 2 | 2 | completed | True | True | 0.0129 | 698065 | 0.893 | 69.000 |
| 3 | 3 | completed | True | True | 0.0129 | 188820 | 0.888 | 68.900 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p45/optimization_rounds_no_invariants/round1/output_model_new.py). `candidate_input.py` was left unchanged. The final path uses Substrate kernels only for the linear algebra, stages GEMM tiles through LDS, uses `S.amdgpu.raw_buffer_load_x4`, and issues `S.amdgpu.mfma_32x32x8_bf16_f32` with a four-wave `2 x 2` warp layout. I also added parameter caching so `forward()` stays cudagraph-safe without rebuilding device metadata tensors each call. Validation passed with the exact required command. The latest round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p45/optimization_rounds_no_invariants/round1/debug_eval.jsonl) shows `correctness: true` and candidate runtime `65.8 us`.
- total_tokens: 520839
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 update in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p45/optimization_rounds_no_invariants/round2/output_model_new.py). Both GEMM kernels now use a double-buffered LDS staging scheme, unroll the K loop by 2, and keep MFMA issue points in the pipelined loop while overlapping the next global loads with current-tile compute. Validation passed with the exact required command. The recorded result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p45/optimization_rounds_no_invariants/round2/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and candidate `runtime_us: 70.1`.
- total_tokens: 698065
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-3 change in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p45/optimization_rounds_no_invariants/round3/output_model_new.py): I removed the explicit K-tail prefetch guards in both GEMM kernels and now rely on the AMD buffer resource `range` behavior so OOB `raw_buffer_load_x4` zero-fills safely. Validation passed with the required command. The round-local eval in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p45/optimization_rounds_no_invariants/round3/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 69.7`.
- total_tokens: 188820
