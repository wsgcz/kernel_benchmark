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
| 1 | 1 | completed | True | True | 0.0217 | 576378 | 0.340 | 15.700 |
| 2 | 2 | completed | True | True | 0.0227 | 552678 | 0.338 | 14.900 |
| 3 | 3 | completed | True | True | 0.0228 | 363187 | 0.335 | 14.700 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the standalone fused MFMA kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p29/optimization_rounds/round1/output_model_new.py). It uses `S.amdgpu.mfma_32x32x8_bf16_f32`, stages operands through LDS, uses `S.amdgpu.raw_buffer_load_x4` for 16-byte global loads, keeps the fixed accumulator writeback mapping, and scales to a 2x2 warp grid. Validation passed with the required command. The latest round-local eval in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p29/optimization_rounds/round1/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 15.6`.
- total_tokens: 576378
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p29/optimization_rounds/round2/output_model_new.py). The kernel now uses a double-buffered, software-pipelined K loop with 2-tile unrolling around `S.amdgpu.mfma_32x32x8_bf16_f32`, keeps the required MFMA/output mapping, and preserves the cudagraph-safe host-side caching. I also resolved the LDS hazard by staging per-warp buffers so the pipeline remains correct under the 2x2 warp-grid execution. Validation passed with the exact required command. The round-local debug result shows `correctness: true` and candidate runtime `14.7 us` in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p29/optimization_rounds/round2/debug_eval.jsonl).
- total_tokens: 552678
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Wrote the final kernel to [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p29/optimization_rounds/round3/output_model_new.py) and left `candidate_input.py` unchanged. Validation passed with the required command. The round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p29/optimization_rounds/round3/debug_eval.jsonl) records `compiled: true` and `correctness: true`. I kept the existing ranged `raw_buffer_load_x4` path intact; attempting to force ranged raw stores for scalar `bf16` outputs failed because `raw_buffer_store_x1` requires `i32` payloads in this Substrate setup, which would need a more invasive packing rewrite.
- total_tokens: 363187
