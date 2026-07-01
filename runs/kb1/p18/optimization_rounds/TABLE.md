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
| 1 | 1 | completed | True | True | 0.0284 | 991386 | 0.232 | 8.170 |
| 2 | 2 | completed | True | True | 0.0309 | 560698 | 0.242 | 7.830 |
| 3 | 3 | completed | True | True | 0.0294 | 157096 | 0.241 | 8.190 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized standalone kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p18/optimization_rounds/round1/output_model_new.py). It now uses `S.amdgpu.mfma_32x32x8_bf16_f32`, stages operands through LDS, loads from global with `S.amdgpu.raw_buffer_load_x4`, and scales the MFMA path across a 2x2 wave grid with the required accumulator writeback mapping. Validation passed with the exact round command. The latest debug record in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p18/optimization_rounds/round1/debug_eval.jsonl) shows `correctness: true` and candidate `runtime_us: 7.88`.
- total_tokens: 991386
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the double-buffered, 2x-unrolled MFMA kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p18/optimization_rounds/round2/output_model_new.py). The K loop now keeps two LDS stages alive, snapshots both stage fragments into registers before any overwrite, and interleaves future `raw_buffer_load_x4` loads between MFMA steps while preserving the required MFMA swizzle and output mapping. Validation passed with the exact round command. The harness reported `correctness: true` and `runtime_us: 7.82` in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p18/optimization_rounds/round2/debug_eval.jsonl).
- total_tokens: 560698
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-3 change in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p18/optimization_rounds/round3/output_model_new.py). The K-loop now relies on the existing `make_rsrc(..., range_bytes)` resource ranges for `raw_buffer_load_x4`, so the explicit loop-time boundary branches around the staged prefetches were removed and the double-buffered MFMA pipeline stays intact. Validation passed with the required command. The round-local eval log at [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p18/optimization_rounds/round3/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 7.86`.
- total_tokens: 157096
