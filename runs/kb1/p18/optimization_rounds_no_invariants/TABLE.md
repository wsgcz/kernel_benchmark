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
| 1 | 1 | completed | True | True | 0.0229 | 705055 | 0.229 | 9.990 |
| 2 | 2 | completed | True | True | 0.0144 | 166593 | 0.242 | 16.800 |
| 3 | 3 | completed | True | True | 0.0144 | 159067 | 0.240 | 16.700 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the final kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p18/optimization_rounds_no_invariants/round1/output_model_new.py). It removes the torch matmul fallback, fixes the actual GEMM shape to `2048 x 8192 x 4096`, stages A/B through LDS with `S.amdgpu.raw_buffer_load_x4`, and keeps MFMA instructions issued inside the substrate kernel. Validation passed with the required command. The round-local eval recorded `correctness: true` and `runtime_us: 9.29` in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p18/optimization_rounds_no_invariants/round1/debug_eval.jsonl).
- total_tokens: 705055
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 kernel update in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p18/optimization_rounds_no_invariants/round2/output_model_new.py). The kernel now uses double-buffered LDS pack storage for A/B, preloads two `BLOCK_K` stages, and runs the K loop unrolled by 2 with stage reloads interleaved between the two MFMA-backed consume steps. I also removed the large shared `a_tile`/`b_tile` scratch arrays and reconstruct scalar values directly from the LDS pack layout, which keeps the four-wave MFMA path intact while shrinking shared working sets. Validation passed with the required harness command. The debug result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p18/optimization_rounds_no_invariants/round2/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 16.7`.
- total_tokens: 166593
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-3 optimization in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p18/optimization_rounds_no_invariants/round3/output_model_new.py) by removing the two `next_k* < K` prefetch guards and relying on the existing ranged `S.amdgpu.make_rsrc(..., range_bytes)` behavior for branchless `raw_buffer_load_x4` zero-fill on OOB accesses. Validation passed with the required command. The round-local debug result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p18/optimization_rounds_no_invariants/round3/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and `runtime_us: 16.8`.
- total_tokens: 159067
