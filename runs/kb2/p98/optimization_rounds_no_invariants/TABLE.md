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
| 1 | 1 | completed | True | True | 0.9126 | 1363413 | 0.282 | 0.309 |
| 2 | 2 | eval_failed | True | False | - | 1182326 | 0.308 | - |
| 3 | 3 | eval_failed | True | False | - | 1710972 | 0.275 | - |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p98/optimization_rounds_no_invariants/round1/output_model_new.py) and left `candidate_input.py` untouched. Validation passed with the required round-local command. The latest debug eval in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p98/optimization_rounds_no_invariants/round1/debug_eval.jsonl) reports `correctness: true` with candidate runtime `0.305 us` versus reference `0.274 us`.
- total_tokens: 1363413
### round2
- status: eval_failed
- unlocked hints: 2
- agent: codex
- summary: Implemented the requested kernel changes in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p98/optimization_rounds_no_invariants/round2/output_model_new.py) and kept [candidate_input.py](/workspace/kb_eval_pipeline/runs/kb2/p98/optimization_rounds_no_invariants/round2/candidate_input.py) unchanged. I also ran the exact round evaluation command. The kernel compiles and runs, but it does not pass correctness yet. The best validated variant in the file currently ends with an output mismatch in `debug_eval.jsonl`, with the latest better run showing `max_difference=1.78125` and `avg_difference=1.085938`.
- total_tokens: 1182326
- error: Output mismatch
### round3
- status: eval_failed
- unlocked hints: 3
- agent: codex
- summary: `output_model_new.py` is restored to the exact starting kernel from `candidate_input.py`. I also ran the required round-local evaluation command. The result shows the baseline kernel itself is not correctness-passing in this harness: - `correctness: false` - `max_difference: 1.781250` - `avg_difference: 1.085938` Because the starting kernel already fails the required correctness check, I did not leave an unvalidated branch-removal variant in place. The current `output_model_new.py` is the safest state I could verify under the round constraints.
- total_tokens: 1710972
- error: Output mismatch
