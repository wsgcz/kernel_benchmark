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
| 1 | 1 | completed | True | True | 0.0307 | 623418 | 0.269 | 8.770 |
| 2 | 2 | completed | True | True | 0.0271 | 154367 | 0.263 | 9.720 |
| 3 | 3 | completed | True | True | 0.0296 | 510119 | 0.270 | 9.110 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the final kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p63/optimization_rounds/round1/output_model_new.py) without modifying `candidate_input.py`. Validation passed with the required harness command. The round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p63/optimization_rounds/round1/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and `runtime_us: 9.03`.
- total_tokens: 623418
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 change in [`output_model_new.py`](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p63/optimization_rounds/round2/output_model_new.py). The K loop is now structured as a double-buffered software pipeline with 16-byte `raw_buffer_load_x4` staging through LDS, two-stage unrolling (`BLOCK_K * 2`), and interleaved prefetch/compute ordering. The wrapper and cached tensor behavior remain cudagraph-safe. Validation passed with the exact required command. The round-local debug result in [`debug_eval.jsonl`](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p63/optimization_rounds/round2/debug_eval.jsonl) shows `compiled: true` and `correctness: true`.
- total_tokens: 154367
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-local kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p63/optimization_rounds/round3/output_model_new.py). The change keeps the existing pipelined MFMA structure and moves the bias path to ranged `raw_buffer_load_x2`, removing direct bias indexing from the epilogue while preserving correctness. Validation passed with the required command. The recorded result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p63/optimization_rounds/round3/debug_eval.jsonl) shows `compiled: true` and `correctness: true`.
- total_tokens: 510119
