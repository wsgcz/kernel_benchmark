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
| 1 | 1 | eval_failed | True | False | - | 1886683 | 0.347 | - |
| 2 | 2 | eval_failed | True | False | - | 2518270 | 0.346 | - |
| 3 | 3 | completed | True | True | 0.9885 | 516803 | 0.344 | 0.348 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: eval_failed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p30/optimization_rounds/round1/output_model_new.py). I also ran the required evaluation command. The kernel compiles and runs, but it does not pass correctness yet. The latest round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p30/optimization_rounds/round1/debug_eval.jsonl) reports `correctness: false` with `max_difference: 4.000000` and `avg_difference: 1.070312`. The remaining issue is in the MFMA operand packing/swizzle path, not in compilation.
- total_tokens: 1886683
- error: Output mismatch
### round2
- status: eval_failed
- unlocked hints: 2
- agent: codex
- summary: Updated [/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p30/optimization_rounds/round2/output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p30/optimization_rounds/round2/output_model_new.py). The file now contains the prompt-aligned MFMA path with a two-tile unrolled loop, two LDS staging buffers, and a corrected `A` repack attempt. `python -m py_compile` passes on the final file. I was not able to get a correctness-passing harness result for this round; the remaining failure is in the GEMM mapping itself, which I reproduced locally against `nn.Linear`.
- total_tokens: 2518270
- error: Output mismatch
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the final model in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p30/optimization_rounds/round3/output_model_new.py). Validation passed with the required command. The round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p30/optimization_rounds/round3/debug_eval.jsonl) shows `correctness: true`, candidate `runtime_us: 0.345`, reference `runtime_us: 0.346`.
- total_tokens: 516803
