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
| 1 | 1 | eval_failed | True | False | - | 2171313 | 0.328 | - |
| 2 | 2 | eval_failed | True | False | - | 1594805 | 0.323 | - |
| 3 | 3 | completed | True | True | 1.0000 | 547200 | 0.329 | 0.329 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: eval_failed
- unlocked hints: 1
- agent: codex
- summary: Implemented an MFMA-based substrate kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p81/optimization_rounds_no_invariants/round1/output_model_new.py). I also ran the required round-local evaluation command. The current kernel compiles and runs, but it does not pass correctness yet. The latest debug eval in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p81/optimization_rounds_no_invariants/round1/debug_eval.jsonl) reports `correctness: false` with max difference `0.574219` and average difference `0.065430`.
- total_tokens: 2171313
- error: Output mismatch
### round2
- status: eval_failed
- unlocked hints: 2
- agent: codex
- summary: Implemented a 2-stage, double-buffered version in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p81/optimization_rounds_no_invariants/round2/output_model_new.py). I also ran the required round-local evaluation command and local reference checks. The kernel still does not pass correctness for this round: the harness reports output mismatch with `max_difference` about `0.6328125` and `avg_difference` about `0.06543`. The debug output is in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p81/optimization_rounds_no_invariants/round2/debug_eval.jsonl). I did not modify `candidate_input.py`.
- total_tokens: 1594805
- error: Output mismatch
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the final round output in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p81/optimization_rounds_no_invariants/round3/output_model_new.py) without modifying [candidate_input.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p81/optimization_rounds_no_invariants/round3/candidate_input.py:1). Validation passed with the required command. The round-local `debug_eval.jsonl` reports `correctness: true`, candidate runtime `0.329 us`, and reference runtime `0.327 us`.
- total_tokens: 547200
