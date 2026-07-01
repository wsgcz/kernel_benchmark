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
| 1 | 1 | completed | True | True | 1.0472 | 430212 | 0.333 | 0.318 |
| 2 | 2 | completed | True | True | 1.0061 | 737864 | 0.331 | 0.329 |
| 3 | 3 | completed | True | True | 1.0119 | 292536 | 0.339 | 0.335 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the final kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p97/optimization_rounds_no_invariants/round1/output_model_new.py) and left `candidate_input.py` unchanged. Validation passed with the exact required command. The round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p97/optimization_rounds_no_invariants/round1/debug_eval.jsonl) reports `compiled: true` and `correctness: true`, with measured runtime `0.327 us` vs reference `0.334 us`.
- total_tokens: 430212
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the updated kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p97/optimization_rounds_no_invariants/round2/output_model_new.py). The substrate path now uses a 4-wave MFMA touch kernel with double-buffered LDS staging and a K-loop unrolled by 2. Validation passed with the required command. `debug_eval.jsonl` reports `correctness: true` and a candidate runtime of `0.326 us` versus `0.331 us` reference.
- total_tokens: 737864
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round output in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p97/optimization_rounds_no_invariants/round3/output_model_new.py). Validation passed with the required command and recorded `correctness: true` in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p97/optimization_rounds_no_invariants/round3/debug_eval.jsonl). The substrate build in this workspace does not lower `raw_buffer_store_x4`, so the final kernel keeps the existing range-backed `raw_buffer_load_x4` path and the proven scalar scratch writeback to preserve correctness.
- total_tokens: 292536
