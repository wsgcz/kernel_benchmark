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
| 1 | 1 | eval_failed | True | False | - | 1581334 | 0.266 | - |
| 2 | 2 | eval_failed | True | False | - | 1414323 | 0.272 | - |
| 3 | 3 | eval_failed | True | False | - | 1811504 | 0.272 | - |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: eval_failed
- unlocked hints: 1
- agent: codex
- summary: Implemented an MFMA/LDS/vector-load kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p14/optimization_rounds_no_invariants/round1/output_model_new.py), but I did not get it to correctness-pass. I validated with the required command and the latest result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p14/optimization_rounds_no_invariants/round1/debug_eval.jsonl) is still failing with `correctness: false`, `max_difference: 32.0`, `avg_difference: 5.34375` on the last run.
- total_tokens: 1581334
- error: Output mismatch
### round2
- status: eval_failed
- unlocked hints: 2
- agent: codex
- summary: Implemented and validated multiple focused variants in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p14/optimization_rounds_no_invariants/round2/output_model_new.py), but I did not reach a correctness-passing result for this round. The key finding is that the starting kernel itself is not correctness-clean under the required harness command: restoring the original `candidate_input.py` logic into `output_model_new.py` still failed with `max_difference=32` and `avg_difference=5.34375` in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p14/optimization_rounds_no_invariants/round2/debug_eval.jsonl). I then tried the software-pipelined double-buffered variant, output-dtype alignment, BF16-only staged reductions, and per-256-row rounded accumulation; none passed. The current file contains the last narrowed attempt. The required evaluation command was run exactly, and the latest recorded result is still `correctness: false` in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p14/optimization_rounds_no_invariants/round2/debug_eval.jsonl).
- total_tokens: 1414323
- error: Output mismatch
### round3
- status: eval_failed
- unlocked hints: 3
- agent: codex
- summary: Wrote the final kernel to [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p14/optimization_rounds_no_invariants/round3/output_model_new.py) and left [candidate_input.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p14/optimization_rounds_no_invariants/round3/candidate_input.py) unchanged. The main changes were: - tightened the raw-buffer load path around subview-backed descriptors so the load-side range comes from `make_rsrc` on the relevant view, - switched the scalar reduction to use `W_SUM_F32`, - removed the per-tile bf16 truncation in the accumulator. I also ran the required eval command and wrote results to [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p14/optimization_rounds_no_invariants/round3/debug_eval.jsonl). The final run compiled but did not pass correctness: `max_difference=32.0`, `avg_difference=5.3125`, `runtime_us=null`. Under the file-access restrictions, I could not safely infer the remaining benchmark-semantic mismatch from local evidence alone.
- total_tokens: 1811504
- error: Output mismatch
