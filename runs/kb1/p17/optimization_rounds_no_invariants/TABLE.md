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
| 1 | 1 | completed | True | True | 0.0316 | 1131169 | 0.242 | 7.650 |
| 2 | 2 | completed | True | True | 0.0319 | 509308 | 0.246 | 7.710 |
| 3 | 3 | completed | True | True | 0.0296 | 226215 | 0.239 | 8.080 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p17/optimization_rounds_no_invariants/round1/output_model_new.py). It uses a 4-wave `64x64x16` tiled GEMM, stages A and B through LDS, uses global `S.amdgpu.raw_buffer_load_x4` loads, and issues `S.amdgpu.mfma_32x32x8_bf16_f32` twice per K-tile in natural half order. Validation passed with the required command. The round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p17/optimization_rounds_no_invariants/round1/debug_eval.jsonl) shows `compiled: true` and `correctness: true`.
- total_tokens: 1131169
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the pipelined MFMA kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p17/optimization_rounds_no_invariants/round2/output_model_new.py). The K loop is now unrolled by 2 with two LDS stages preloaded and then refilled in a fixed ping-pong schedule between the two MFMA steps, preserving the existing 4-wave `2 x 2` warp-grid mapping and MFMA operand layout. Validation passed with the exact required command. The round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p17/optimization_rounds_no_invariants/round2/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 7.65`.
- total_tokens: 509308
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round output in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p17/optimization_rounds_no_invariants/round3/output_model_new.py). The kernel keeps the existing pipelined/double-buffered MFMA structure and uses AMD buffer resources with byte ranges for the global `raw_buffer_load_x4` paths. Validation passed with the required harness command. The recorded result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p17/optimization_rounds_no_invariants/round3/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and `runtime_us: 7.66`. One concrete constraint in this Substrate build: `S.amdgpu.raw_buffer_store_x1` only accepts 32-bit data, so bf16 scalar writeback cannot be legally converted to raw buffer stores without a larger writeback reorganization. I therefore left the final bf16 tensor stores unchanged to preserve a correctness-passing kernel.
- total_tokens: 226215
