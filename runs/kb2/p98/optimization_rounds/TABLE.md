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
| 1 | 1 | completed | True | True | 0.0015 | 1534168 | 0.278 | 180.000 |
| 2 | 2 | completed | True | True | 0.0015 | 347274 | 0.274 | 184.000 |
| 3 | 3 | completed | True | True | 0.0016 | 182815 | 0.279 | 180.000 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized MFMA-based fused kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p98/optimization_rounds/round1/output_model_new.py). It now uses 4 waves as a 2x2 warp grid, stages operands through LDS, issues `S.amdgpu.mfma_32x32x8_bf16_f32`, uses `S.amdgpu.raw_buffer_load_x4`, and keeps the fused pool/GELU/max reduction in the kernel. I also made the weight/bias path cudagraph-safe by caching the device/dtype-converted tensors and only rebuilding on storage-pointer changes. Validation passed with the required harness command. The latest round-local result in `debug_eval.jsonl` reports `correctness: true` and `runtime_us: 180.0`.
- total_tokens: 1534168
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 optimization in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p98/optimization_rounds/round2/output_model_new.py). The K path now uses ping-pong LDS buffers for A/B, processes two `BLOCK_K` tiles per loop iteration, and interleaves next-tile `raw_buffer_load_x4` + LDS staging between the two natural `mfma_32x32x8_bf16_f32` steps while keeping the required MFMA swizzle and C writeback mapping unchanged. Validation passed with the exact required command. The round-local eval in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p98/optimization_rounds/round2/debug_eval.jsonl) reports `compiled: true` and `correctness: true`.
- total_tokens: 347274
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-specific change in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p98/optimization_rounds/round3/output_model_new.py): the pipelined K-loop now always issues the `raw_buffer_load_x4` prefetches for `next_even` and `next_odd`, relying on the resource `range` in `make_rsrc(...)` for safe zero-fill on OOB reads instead of explicit loop-guard branches. Validation passed with the required command. The round-local eval output in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p98/optimization_rounds/round3/debug_eval.jsonl) shows `compiled: true` and `correctness: true`.
- total_tokens: 182815
