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
| 1 | 1 | completed | True | True | 2.7630 | 280955 | 0.218 | 0.079 |
| 2 | 2 | completed | True | True | 2.2830 | 560579 | 0.213 | 0.093 |
| 3 | 3 | completed | True | True | 1.9273 | 152391 | 0.212 | 0.110 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p12/optimization_rounds/round1/output_model_new.py). I left [candidate_input.py](/workspace/kb_eval_pipeline/runs/kb1/p12/optimization_rounds/round1/candidate_input.py) unchanged. Validation passed with the exact required command. The round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p12/optimization_rounds/round1/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and `runtime_us: 0.0796` versus `ref_runtime_us: 0.214`.
- total_tokens: 280955
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p12/optimization_rounds/round2/output_model_new.py). The rewrite keeps the candidate’s known-good execution shape, switches the row processing to a double-buffered `16 x 64` LDS pipeline, uses vectorized `raw_buffer_load_x4` staging for both operands, and issues two MFMA steps per pipeline body. Validation passed with the exact required command. The round-local debug eval reports `compiled: true`, `correctness: true`, and candidate `runtime_us: 0.0935` on `gfx942` in `/workspace/kb_eval_pipeline/runs/kb1/p12/optimization_rounds/round2/debug_eval.jsonl`.
- total_tokens: 560579
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-3 change in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p12/optimization_rounds/round3/output_model_new.py): the K-loop prefetch path now relies on the raw buffer resource `range` for OOB-safe `raw_buffer_load_x4` behavior, removing the explicit `next_chunk < ROW_CHUNKS` guard around the next-stage global loads. I left `candidate_input.py` unchanged. Validation passed with the required command. The round-local debug eval in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p12/optimization_rounds/round3/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 0.111` versus `ref_runtime_us: 0.209`.
- total_tokens: 152391
