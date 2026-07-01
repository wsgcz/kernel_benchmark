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
| 1 | 1 | completed | True | True | 0.0002 | 594397 | 0.894 | 3950.000 |
| 2 | 2 | completed | True | True | 0.0002 | 1351549 | 0.772 | 3930.000 |
| 3 | 3 | completed | True | True | 0.0004 | 411791 | 0.768 | 1730.000 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p55/optimization_rounds_no_invariants/round1/output_model_new.py). Validation with the exact required command passed correctness. The round-local eval in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p55/optimization_rounds_no_invariants/round1/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 3950.0`.
- total_tokens: 594397
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Updated [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p55/optimization_rounds_no_invariants/round2/output_model_new.py) only. The file currently keeps the original fused kernel body and adds cudagraph-safe caching for the transposed weight and bias so `forward()` no longer rebuilds those tensors unless the underlying parameter storage pointer changes. I was not able to complete a correctness-passing validation for this round. The required evaluator command stalled repeatedly before producing a result after the later edits, so there is no confirmed passing `debug_eval.jsonl` for the final file.
- total_tokens: 1351549
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-3 change in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p55/optimization_rounds_no_invariants/round3/output_model_new.py): the `raw_buffer_load_x4` paths now pass explicit byte `range` values and split row-base bytes from intra-row bytes, so tail accesses can zero-fill without separate OOB guards. Validation passed with the required command. The round-local eval in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p55/optimization_rounds_no_invariants/round3/debug_eval.jsonl) reports `compiled: true` and `correctness: true` on `gfx942`.
- total_tokens: 411791
