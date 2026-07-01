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
| 1 | 1 | completed | True | True | 1.0063 | 1146916 | 0.318 | 0.316 |
| 2 | 2 | completed | True | True | 0.9483 | 451959 | 0.312 | 0.329 |
| 3 | 3 | completed | True | True | 0.9637 | 333196 | 0.319 | 0.331 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized submission in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p84/optimization_rounds_no_invariants/round1/output_model_new.py). Validation passed with the required command. The round-local eval in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p84/optimization_rounds_no_invariants/round1/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and candidate runtime `0.316 us` versus reference `0.338 us`.
- total_tokens: 1146916
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p84/optimization_rounds_no_invariants/round2/output_model_new.py). The kernel now uses a four-wave `2 x 2` warp grid, stages A/B through LDS with `raw_buffer_load_x4`, and runs a double-buffered MFMA pipeline with the K path unrolled by 2 across four `16`-wide K tiles. Validation passed with the required command. The harness result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p84/optimization_rounds_no_invariants/round2/debug_eval.jsonl) reports `compiled: true` and `correctness: true`.
- total_tokens: 451959
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-3 change in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p84/optimization_rounds_no_invariants/round3/output_model_new.py). The kernel now makes the A/B buffer ranges explicit in bytes and keeps the branch-free `raw_buffer_load_x4` path aligned with range-based OOB handling. Validation passed with the required command. The round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p84/optimization_rounds_no_invariants/round3/debug_eval.jsonl) shows `compiled: true` and `correctness: true`.
- total_tokens: 333196
