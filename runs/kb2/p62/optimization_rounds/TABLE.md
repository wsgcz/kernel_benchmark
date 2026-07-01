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
| 1 | 1 | eval_failed | True | False | - | 1403056 | 0.574 | - |
| 2 | 2 | eval_failed | True | False | - | 3006455 | 0.570 | - |
| 3 | 3 | completed | True | True | 0.9965 | 1428958 | 0.570 | 0.572 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: eval_failed
- unlocked hints: 1
- agent: codex
- summary: Implemented the MFMA/raw-buffer/LDS rewrite in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p62/optimization_rounds/round1/output_model_new.py), keeping `candidate_input.py` unchanged. Current status: it compiles, launches, and the round-local debug eval was rerun, but it is still not correctness-passing. The latest harness result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p62/optimization_rounds/round1/debug_eval.jsonl) reports `correctness: false` with `max_difference: 6.968750` and `avg_difference: 1.023438`. The remaining issue is the MFMA operand/output swizzle, not a compile/runtime failure.
- total_tokens: 1403056
- error: Output mismatch
### round2
- status: eval_failed
- unlocked hints: 2
- agent: codex
- summary: Updated [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p62/optimization_rounds/round2/output_model_new.py) with the software-pipelined double-buffered MFMA GEMM path using LDS staging and K-loop unrolling by 2. I also ran the required round-local evaluation command. The kernel compiles and runs, but correctness is still failing in the harness: the latest `debug_eval.jsonl` entry reports `correctness: false` with `max_difference: 7.031250` and `avg_difference: 0.921875`.
- total_tokens: 3006455
- error: Output mismatch
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round output in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p62/optimization_rounds/round3/output_model_new.py). The file includes the ranged AMD buffer-load GEMM kernel work, and I kept all edits confined to that output file. Validation passed with the required command. The latest round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p62/optimization_rounds/round3/debug_eval.jsonl) reports `correctness: true`, candidate `runtime_us: 0.572`, reference `ref_runtime_us: 0.566`.
- total_tokens: 1428958
