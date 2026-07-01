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
| 1 | 1 | completed | True | True | 0.0359 | 259158 | 0.277 | 7.720 |
| 2 | 2 | completed | True | True | 0.0360 | 157259 | 0.276 | 7.660 |
| 3 | 3 | completed | True | True | 0.0376 | 362826 | 0.279 | 7.430 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p99/optimization_rounds/round1/output_model_new.py). It now uses a standalone AMD MFMA GEMM path with `S.amdgpu.mfma_32x32x8_bf16_f32`, stages operands through LDS, uses `S.amdgpu.raw_buffer_load_x4` for 16-byte global loads, scales the GEMM tile to 4 waves as a `2 x 2` warp grid, and keeps the post-GEMM GELU and softmax in substrate kernels. I also added simple parameter caching so `forward()` does not rebuild transient metadata-style tensors each call unless the source storage pointer changes. Validation passed with the exact required command. The round-local debug result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p99/optimization_rounds/round1/debug_eval.jsonl) reports `compiled: true` and `correctness: true`.
- total_tokens: 259158
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the next-round optimization in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p99/optimization_rounds/round2/output_model_new.py). The GEMM path now uses double-buffered LDS staging for A/B, processes K in pairs (`16 + 16`) with an unrolled-by-2 loop structure, and interleaves MFMA with the next buffer refill while preserving the required MFMA swizzle and accumulator mapping. Validation passed with the exact required command. The round-local debug result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p99/optimization_rounds/round2/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 7.9`.
- total_tokens: 157259
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Wrote the final kernel to [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p99/optimization_rounds/round3/output_model_new.py) and left [candidate_input.py](/workspace/kb_eval_pipeline/runs/kb2/p99/optimization_rounds/round3/candidate_input.py) unchanged. Validation passed with the required command. The recorded result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p99/optimization_rounds/round3/debug_eval.jsonl) shows `correctness: true` and `runtime_us: 7.43`.
- total_tokens: 362826
