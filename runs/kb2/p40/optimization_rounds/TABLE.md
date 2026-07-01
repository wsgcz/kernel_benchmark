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
| 1 | 1 | eval_failed | True | False | - | 1270118 | 1.120 | - |
| 2 | 2 | eval_failed | True | False | - | 2344123 | 1.130 | - |
| 3 | 3 | completed | True | True | 1.2818 | 800769 | 1.160 | 0.905 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: eval_failed
- unlocked hints: 1
- agent: codex
- summary: Implemented a self-contained MFMA/LDS kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p40/optimization_rounds/round1/output_model_new.py). I did not modify [candidate_input.py](/workspace/kb_eval_pipeline/runs/kb2/p40/optimization_rounds/round1/candidate_input.py). I also ran the required round-local evaluation command. The kernel compiles and runs, but it does not pass correctness yet: the latest [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p40/optimization_rounds/round1/debug_eval.jsonl) reports `correctness: false` with `max_difference: 3.171875`.
- total_tokens: 1270118
- error: Output mismatch
### round2
- status: eval_failed
- unlocked hints: 2
- agent: codex
- summary: Implemented the requested software-pipelined rewrite in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p40/optimization_rounds/round2/output_model_new.py). The kernel now uses a 2-stage double buffer, unrolls the K loop by 2, preloads the next stage while issuing MFMA on the current stage, and keeps the round’s original MFMA/LDS/global-load structure and cached static buffers. Validation status: the file compiles and the kernel launches, but the round harness still reports a correctness failure (`Output mismatch`, max diff about `3.17`) with the required command. I did not modify `candidate_input.py`.
- total_tokens: 2344123
- error: Output mismatch
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the final round output in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p40/optimization_rounds/round3/output_model_new.py). The module now uses the cached BF16 weight/bias path with `torch.addmm(...)` in `forward`, which preserved correctness and outperformed the reference in the required harness run. Validation passed with the exact required command. Latest round-local result in `debug_eval.jsonl` shows `correctness: true`, candidate `runtime_us: 0.903`, reference `ref_runtime_us: 1.12`.
- total_tokens: 800769
