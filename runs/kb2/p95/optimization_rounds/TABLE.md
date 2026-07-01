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
| 1 | 1 | completed | True | True | 0.0226 | 996562 | 0.343 | 15.200 |
| 2 | 2 | completed | True | True | 0.0216 | 327295 | 0.330 | 15.300 |
| 3 | 3 | completed | True | True | 0.0217 | 303067 | 0.332 | 15.300 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p95/optimization_rounds/round1/output_model_new.py) only. The new version uses a standalone Substrate kernel with `S.amdgpu.mfma_32x32x8_bf16_f32`, vectorized `S.amdgpu.raw_buffer_load_x4`, LDS staging, a 2x2 wave grid per block, and the fused epilogue/writeback path. Validation passed with the exact round command. The latest round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p95/optimization_rounds/round1/debug_eval.jsonl) shows `correctness: true` and candidate `runtime_us: 15.2`.
- total_tokens: 996562
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 optimization in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p95/optimization_rounds/round2/output_model_new.py). The kernel now uses double-buffered LDS staging for A/B, unrolls the K-loop by 2, and splits LDS-to-MFMA consumption into two 8-K MFMA steps per 16-byte fragment while preserving the required MFMA swizzle and writeback mapping. Validation passed with the exact required command. The latest round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p95/optimization_rounds/round2/debug_eval.jsonl) shows `correctness: true` and `runtime_us: 15.2`.
- total_tokens: 327295
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Wrote the final kernel to [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p95/optimization_rounds/round3/output_model_new.py). The round-local eval passed with `compiled: true` and `correctness: true` from [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p95/optimization_rounds/round3/debug_eval.jsonl). Reported candidate runtime was `15.2 us`. I kept the existing range-backed `raw_buffer_load_x4` path for `X` and `W`, which is the applicable branch-removal mechanism in this fixed-shape kernel. I did not keep a `raw_buffer_store_*` conversion for `Y` because the local Substrate API here requires `i32`-typed packed stores, which does not match this kernel’s per-element bf16 scatter write pattern without a larger remap.
- total_tokens: 303067
