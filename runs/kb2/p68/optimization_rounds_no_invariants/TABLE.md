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
| 1 | 1 | completed | True | True | - | 1229476 | 0.237 | - |
| 2 | 2 | completed | True | True | 0.0019 | 980047 | 0.237 | 128.000 |
| 3 | 3 | completed | True | True | 0.0018 | 493316 | 0.234 | 128.000 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p68/optimization_rounds_no_invariants/round1/output_model_new.py) and left `candidate_input.py` unchanged. Validation with the required harness command passed correctness for this round: `correctness: true (1 / 1)`. The debug eval also reported a candidate profiling failure under CUDA graph capture (`HIP error: operation not permitted when stream is capturing`), so correctness is confirmed but performance timing was not successfully captured in that run.
- total_tokens: 1229476
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the final kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p68/optimization_rounds_no_invariants/round2/output_model_new.py) only. The kernel now keeps the MFMA path present, stages `A` and `B` through double-buffered LDS, unrolls the K loop by 2, and preserves correctness with a Substrate-safe pipeline structure. I also fixed the forward-path cache behavior so cudagraph profiling works without rebuilding or copying cached device tensors on every call. Validation used the exact round command and passed correctness with cudagraph timing. The latest round-local result in `debug_eval.jsonl` shows `correctness: true` and `runtime_us: 127.0`.
- total_tokens: 980047
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the final kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p68/optimization_rounds_no_invariants/round3/output_model_new.py). The hot-path change is focused: I removed the redundant `local_row < BLOCK_M` guards from the K-tile accumulation while preserving the existing pipelined MFMA/LDS structure and leaving the raw `x_rsrc`/`w_rsrc` range-backed loads intact. Validation passed with the required command. The round-local debug result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p68/optimization_rounds_no_invariants/round3/debug_eval.jsonl) reports `correctness: true` and `runtime_us: 129.0`. `candidate_input.py` was not modified.
- total_tokens: 493316
