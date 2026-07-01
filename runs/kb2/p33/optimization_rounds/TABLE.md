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
| 1 | 1 | eval_failed | True | False | - | 1532959 | 0.293 | - |
| 2 | 2 | eval_failed | True | False | - | 1742904 | 0.295 | - |
| 3 | 3 | eval_failed | True | False | - | 468597 | 0.295 | - |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: eval_failed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p33/optimization_rounds/round1/output_model_new.py). I did not modify [candidate_input.py](/workspace/kb_eval_pipeline/runs/kb2/p33/optimization_rounds/round1/candidate_input.py). Validation with the required harness command compiled and ran the MFMA path, but it did not reach a correctness pass. The closest run was `max_difference=0.0625`, `avg_difference=0.002365`, with the result recorded in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p33/optimization_rounds/round1/debug_eval.jsonl).
- total_tokens: 1532959
- error: Output mismatch
### round2
- status: eval_failed
- unlocked hints: 2
- agent: codex
- summary: Implemented the kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p33/optimization_rounds/round2/output_model_new.py). Validation with the required command still does not pass correctness. The best result I reached in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p33/optimization_rounds/round2/debug_eval.jsonl) is `correctness: false` with `max_difference: 0.054688` and `avg_difference: 0.000001`. I also verified the starting kernel itself fails the same evaluator, so the remaining issue is not just the pipelining change.
- total_tokens: 1742904
- error: Output mismatch
### round3
- status: eval_failed
- unlocked hints: 3
- agent: codex
- summary: `output_model_new.py` is currently restored to the exact starting kernel from [candidate_input.py](/workspace/kb_eval_pipeline/runs/kb2/p33/optimization_rounds/round3/candidate_input.py) to avoid leaving a known-worse variant in place. I ran the required round-local evaluation command exactly as specified. It compiled, but the baseline itself does not pass correctness on this case: `correctness=false`, `max_difference=0.054688`, from [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p33/optimization_rounds/round3/debug_eval.jsonl). The branch-removal attempt also failed correctness, so I reverted it. I did not modify `candidate_input.py`.
- total_tokens: 468597
- error: Output mismatch
