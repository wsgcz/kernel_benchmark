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
| 1 | 1 | completed | True | True | 0.0156 | 1814434 | 0.231 | 14.800 |
| 2 | 2 | completed | True | True | 0.0161 | 685813 | 0.232 | 14.400 |
| 3 | 3 | completed | True | True | 0.0161 | 273335 | 0.232 | 14.400 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p01/optimization_rounds/round1/output_model_new.py). It uses `S.amdgpu.mfma_32x32x8_bf16_f32`, stages operands through LDS, loads from global with `S.amdgpu.raw_buffer_load_x4`, and scales the computation to a 2 x 2 wave grid over a 64 x 64 block while preserving the per-wave MFMA/writeback invariants. Validation passed with the required round-local command. The latest `debug_eval.jsonl` reports `correctness: true` with candidate `runtime_us: 14.7`. `candidate_input.py` was not modified.
- total_tokens: 1814434
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p01/optimization_rounds/round2/output_model_new.py). The kernel keeps the required 4-wave MFMA mapping, stages A/B through LDS, and now uses a 2-tile software pipeline with double buffering and a K-loop unrolled by 2. Validation passed with the exact required command. The round-local debug result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p01/optimization_rounds/round2/debug_eval.jsonl) shows `correctness: true` and `runtime_us: 14.4`.
- total_tokens: 685813
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the final kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p01/optimization_rounds/round3/output_model_new.py) without modifying the input file. Validation passed with the required command. The latest round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p01/optimization_rounds/round3/debug_eval.jsonl) shows `correctness: true` and candidate runtime `14.7 us`. One concrete note: I attempted the output-side `raw_buffer_store_*` conversion, but this Substrate path only accepted packed `i32` payloads and did not support the kernel’s per-lane scalar `bf16` output layout without a larger rewrite. I kept the existing ranged raw-buffer load path and restored the known-correct store path to finish the round with a correctness-passing kernel.
- total_tokens: 273335
