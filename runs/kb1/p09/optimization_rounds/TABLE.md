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
| 1 | 1 | completed | True | True | 0.1050 | 876329 | 0.739 | 7.040 |
| 2 | 2 | completed | True | True | 0.1057 | 308792 | 0.744 | 7.040 |
| 3 | 3 | completed | True | True | 0.1064 | 220994 | 0.749 | 7.040 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the MFMA-based Substrate kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p09/optimization_rounds/round1/output_model_new.py). It now uses `S.amdgpu.mfma_32x32x8_bf16_f32`, global `raw_buffer_load_x4`, LDS staging, and a 4-wave `2 x 2` warp grid with the required accumulator writeback mapping. Validation passed with the exact round-local command. The latest debug eval in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p09/optimization_rounds/round1/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and candidate `runtime_us: 7.04`.
- total_tokens: 876329
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p09/optimization_rounds/round2/output_model_new.py). The kernel keeps the required 4-wave MFMA mapping and rewrites the fixed `K=32` path as a software-pipelined two-stage schedule: preload the first `K=16` slice into LDS, fetch the second slice from global while the first MFMA step is running, write that second slice into the alternate LDS buffer between the two MFMA instructions, then consume the second buffer after a barrier. That gives a double-buffered, unrolled-by-2 execution without changing the MFMA swizzle or accumulator writeback mapping. Validation passed with the exact required command. The round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p09/optimization_rounds/round2/debug_eval.jsonl) reports `compiled: true` and `correctness: true`.
- total_tokens: 308792
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round output in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p09/optimization_rounds/round3/output_model_new.py). Validation passed with the required command. The recorded result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p09/optimization_rounds/round3/debug_eval.jsonl) shows `compiled: true` and `correctness: true` on `2026-06-25`, with measured candidate runtime `7.04 us`. I kept the existing range-based `raw_buffer_load_x4` path for A/B, which already removes explicit OOB load branches in this kernel. I attempted the same pattern for C stores, but the local substrate backend only accepts `i32` payloads for `raw_buffer_store_x1`, which does not fit this scalar bf16 writeback layout without a larger output repacking change.
- total_tokens: 220994
