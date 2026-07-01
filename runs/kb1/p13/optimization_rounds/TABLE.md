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
| 1 | 1 | completed | True | True | 0.0304 | 554962 | 0.223 | 7.340 |
| 2 | 2 | completed | True | True | 0.0314 | 160319 | 0.230 | 7.330 |
| 3 | 3 | completed | True | True | 0.0284 | 131522 | 0.223 | 7.860 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p13/optimization_rounds/round1/output_model_new.py). It now uses `S.amdgpu.mfma_32x32x8_bf16_f32`, stages operands through LDS, loads global data with `S.amdgpu.raw_buffer_load_x4`, scales to a `2 x 2` wave grid per block, and writes back `C` with the required fixed accumulator mapping. Validation passed with the required harness command. The recorded result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p13/optimization_rounds/round1/debug_eval.jsonl) shows `correctness: true` and `runtime_us: 7.34`.
- total_tokens: 554962
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 optimization in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p13/optimization_rounds/round2/output_model_new.py). The kernel now uses two LDS buffers, unrolls the K-loop by 2, and issues the next `raw_buffer_load_x4` refills between the two natural BF16 MFMA halves for each staged tile while keeping the required 4-wave, 2x2 warp-grid mapping and fixed accumulator writeback layout. Validation passed with the exact required command. The round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p13/optimization_rounds/round2/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 7.33`.
- total_tokens: 160319
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-3 kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p13/optimization_rounds/round3/output_model_new.py). The change removes the `has_next` branch from the pipelined global prefetch path and relies on `S.amdgpu.make_rsrc(..., range_bytes)` plus `raw_buffer_load_x4` zero-fill for speculative K-tail loads. Validation passed with the required harness command. The round-local eval reported `compiled: true`, `correctness: true`, and `runtime_us: 7.48` in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p13/optimization_rounds/round3/debug_eval.jsonl).
- total_tokens: 131522
