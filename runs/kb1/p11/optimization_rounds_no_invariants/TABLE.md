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
| 1 | 1 | completed | True | True | 0.0472 | 716552 | 1.180 | 25.000 |
| 2 | 2 | completed | True | True | 0.0361 | 452139 | 1.190 | 33.000 |
| 3 | 3 | completed | True | True | 0.0329 | 481272 | 1.180 | 35.900 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p11/optimization_rounds_no_invariants/round1/output_model_new.py) and left [candidate_input.py](/workspace/kb_eval_pipeline/runs/kb1/p11/optimization_rounds_no_invariants/round1/candidate_input.py) unchanged. Validation passed with the exact round command. The harness reported `compiled: true`, `correctness: true`, and a candidate runtime of about `25.7 us` with `cudagraph` timing.
- total_tokens: 716552
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p11/optimization_rounds_no_invariants/round2/output_model_new.py). The update adds a two-stage LDS ping-pong buffer, unrolls the K loop by 2, keeps the 4-wave 2x2 warp-grid structure, and preserves MFMA issue points with `raw_buffer_load_x4` staging from global memory. Validation passed with the required harness command. The round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p11/optimization_rounds_no_invariants/round2/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 31.9`.
- total_tokens: 452139
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-3 kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p11/optimization_rounds_no_invariants/round3/output_model_new.py). The key change for this round is that the K-loop preload path now relies on the buffer descriptor `range` for branchless `raw_buffer_load_x4` OOB handling, so the explicit `if k0 + ... < L_DIM` guards were removed. I also switched the output path to raw buffer stores via a descriptor-backed `c_rsrc`, with per-thread bf16 packing through shared `u32` storage. Validation passed with the required command. The round-local eval result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p11/optimization_rounds_no_invariants/round3/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and candidate `runtime_us: 36.5`.
- total_tokens: 481272
