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
| 1 | 1 | completed | True | True | 0.0186 | 325272 | 0.276 | 14.800 |
| 2 | 2 | completed | True | True | 0.0186 | 746419 | 0.276 | 14.800 |
| 3 | 3 | completed | True | True | 0.0186 | 315118 | 0.276 | 14.800 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p76/optimization_rounds/round1/output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p76/optimization_rounds/round1/output_model_new.py:1) only, leaving `candidate_input.py` unchanged. The new kernel uses `S.amdgpu.mfma_32x32x8_bf16_f32`, stages A and B through LDS, loads global data with `S.amdgpu.raw_buffer_load_x4`, and scales the MFMA work across a 2x2 wave grid in a 256-thread block. I also cached the contiguous weight/bias tensors on the Python side to keep the forward path cudagraph-safe. Validation passed with the required command. The round-local debug output at [/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p76/optimization_rounds/round1/debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p76/optimization_rounds/round1/debug_eval.jsonl:1) reports `compiled: true` and `correctness: true`.
- total_tokens: 325272
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p76/optimization_rounds/round2/output_model_new.py) only. The kernel keeps the 4-wave MFMA mapping, stages operands through LDS in split half-fragments, and unrolls the K loop by 2 so each iteration issues four `mfma_32x32x8_bf16_f32` steps from two 16-byte operand chunks per thread. The existing cudagraph-safe weight/bias caching path was preserved. Validation passed with the exact required command. The round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p76/optimization_rounds/round2/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and `runtime_us: 14.8`.
- total_tokens: 746419
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Wrote the final kernel to [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p76/optimization_rounds/round3/output_model_new.py) and left `candidate_input.py` unchanged. Validation passed with the required command. The round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p76/optimization_rounds/round3/debug_eval.jsonl) shows `correctness: true` and `runtime_us: 15.1`. One constraint surfaced during implementation: this Substrate build rejects `S.amdgpu.raw_buffer_store_x1` for `bf16`, so the store-side range conversion was not viable without a larger writeback redesign. I kept the existing pipelined MFMA/LDS structure intact and made the raw-buffer range usage explicit on the load descriptors.
- total_tokens: 315118
