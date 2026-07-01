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
| 1 | 1 | completed | True | True | 0.0096 | 847698 | 0.282 | 29.300 |
| 2 | 2 | completed | True | True | 0.0095 | 930149 | 0.279 | 29.500 |
| 3 | 3 | completed | True | True | 0.0093 | 482926 | 0.287 | 30.800 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p09/optimization_rounds_no_invariants/round1/output_model_new.py). I left `candidate_input.py` unchanged. Validation passed with the exact required command. The round-local harness reported `compiled: true`, `correctness: true`, and `runtime_us: 29.9` in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p09/optimization_rounds_no_invariants/round1/debug_eval.jsonl).
- total_tokens: 847698
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round update in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p09/optimization_rounds_no_invariants/round2/output_model_new.py). The kernel now uses a 2-tile unrolled K-loop with double-buffered LDS staging logic while preserving the numerically stable MFMA path. Validation passed with the required harness command. The latest round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p09/optimization_rounds_no_invariants/round2/debug_eval.jsonl) shows `correctness: true` and `runtime_us: 29.9`.
- total_tokens: 930149
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-local kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p09/optimization_rounds_no_invariants/round3/output_model_new.py). The main change was adding a range-backed raw buffer descriptor for `BIAS` and switching the epilogue bias fetch to `S.amdgpu.raw_buffer_load_x2`, while keeping the existing pipelined MFMA structure intact. Validation passed with the required command. The debug result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p09/optimization_rounds_no_invariants/round3/debug_eval.jsonl) reports `compiled: true` and `correctness: true`.
- total_tokens: 482926
