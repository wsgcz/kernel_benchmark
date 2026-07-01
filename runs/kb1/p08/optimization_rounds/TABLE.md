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
| 1 | 1 | completed | True | True | 0.0175 | 1674078 | 0.549 | 31.400 |
| 2 | 2 | completed | True | True | 0.0363 | 724539 | 0.555 | 15.300 |
| 3 | 3 | completed | True | True | 0.0320 | 326267 | 0.540 | 16.900 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p08/optimization_rounds/round1/output_model_new.py) and left `candidate_input.py` unchanged. Validation passed with the required round-local command. The latest debug eval recorded `correctness: true` and `runtime_us: 31.6` in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p08/optimization_rounds/round1/debug_eval.jsonl).
- total_tokens: 1674078
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p08/optimization_rounds/round2/output_model_new.py) as a standalone MFMA path with double-buffered LDS staging, pair-unrolled K processing, real `S.amdgpu.mfma_32x32x8_bf16_f32` accumulation, vectorized `S.amdgpu.raw_buffer_load_x4` operand loads, and the required 2x2 wave tiling plus fixed accumulator writeback mapping. Validation passed with the exact required command. The round-local debug result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p08/optimization_rounds/round2/debug_eval.jsonl) reports `correctness: true` and candidate `runtime_us: 15.6`.
- total_tokens: 724539
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p08/optimization_rounds/round3/output_model_new.py). The change removes the A/B pad-copy path and switches the kernel to load from the real `(M, K)` and `(K, N)` tensors using `S.amdgpu.make_rsrc(..., range_bytes)` plus range-limited `raw_buffer_load_x4`, so OOB K/N accesses safely read as zero without explicit guards. I kept the padded output buffer path because `raw_buffer_store_x1` in this environment only accepts `i32`, so scalar bf16 range-stores were not compilable. Validation passed with the required command. The round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p08/optimization_rounds/round3/debug_eval.jsonl) reports `"compiled": true` and `"correctness": true`.
- total_tokens: 326267
