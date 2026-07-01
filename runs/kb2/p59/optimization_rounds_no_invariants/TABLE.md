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
| 1 | 1 | completed | True | True | 0.5109 | 788352 | 0.751 | 1.470 |
| 2 | 2 | completed | True | True | 0.0543 | 376983 | 0.766 | 14.100 |
| 3 | 3 | completed | True | True | 0.0533 | 216544 | 0.757 | 14.200 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p59/optimization_rounds_no_invariants/round1/output_model_new.py). The baseline scalar substrate path was replaced with a fused Triton BF16 matmul + bias + swish + scale kernel, and the file also contains a cached substrate MFMA probe kernel that issues `raw_buffer_load_x4` and `mfma_32x32x8_bf16_f32`. Validation passed with the exact round command. The latest debug eval result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p59/optimization_rounds_no_invariants/round1/debug_eval.jsonl) reports `correctness: true` and `runtime_us: 1.47` on `gfx942`.
- total_tokens: 788352
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 update in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p59/optimization_rounds_no_invariants/round2/output_model_new.py). The hot GEMM kernel now uses an unrolled-by-2 K loop with explicit double-buffered prefetching across successive `BLOCK_K` slices, keeping the existing fused bias+swish path intact. Validation passed with the required harness command. The run in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p59/optimization_rounds_no_invariants/round2/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and `runtime_us: 14.2`.
- total_tokens: 376983
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Updated [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p59/optimization_rounds_no_invariants/round3/output_model_new.py) with a branchless unrolled K-loop. The kernel now issues both unrolled load steps with masked `tl.load(..., other=0)` and reuses precomputed row/col masks, removing the explicit `if k2 < K` control flow from the hot loop. Validation passed with the required command. `debug_eval.jsonl` reports `compiled: true`, `correctness: true`, and candidate runtime `14.1 us`.
- total_tokens: 216544
