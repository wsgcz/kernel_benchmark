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
| 1 | 1 | eval_failed | True | False | - | 1213556 | 0.295 | - |
| 2 | 2 | eval_failed | True | False | - | 2360027 | 0.294 | - |
| 3 | 3 | completed | True | True | 0.6433 | 799589 | 0.440 | 0.684 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: eval_failed
- unlocked hints: 1
- agent: codex
- summary: Implemented an MFMA-based substrate kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p33/optimization_rounds_no_invariants/round1/output_model_new.py) and validated it with the required round command. It does not pass correctness yet. The latest round-local eval in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p33/optimization_rounds_no_invariants/round1/debug_eval.jsonl) reports `correctness: false` with `avg_difference: 0.828125` and `max_difference: 6.343750`. The remaining issue is the MFMA/LDS operand or accumulator layout, not compilation.
- total_tokens: 1213556
- error: Output mismatch
### round2
- status: eval_failed
- unlocked hints: 2
- agent: codex
- summary: Implemented the kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p33/optimization_rounds_no_invariants/round2/output_model_new.py). I validated with the required command. The current version still does not pass correctness: the latest harness run in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p33/optimization_rounds_no_invariants/round2/debug_eval.jsonl) reports `correctness: false` with `max_difference: 6.40625` and `avg_difference: 0.824219`.
- total_tokens: 2360027
- error: Output mismatch
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Wrote the final file to [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p33/optimization_rounds_no_invariants/round3/output_model_new.py). Validation passed with the required harness: `correctness: true` in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p33/optimization_rounds_no_invariants/round3/debug_eval.jsonl). The recorded candidate runtime was `0.669 us` versus `0.442 us` for the reference.
- total_tokens: 799589
