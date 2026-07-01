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
| 1 | 1 | completed | True | True | 0.0191 | 1506445 | 0.281 | 14.700 |
| 2 | 2 | completed | True | True | 0.0189 | 179689 | 0.280 | 14.800 |
| 3 | 3 | completed | True | True | 0.0173 | 339232 | 0.273 | 15.800 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p76/optimization_rounds_no_invariants/round1/output_model_new.py). It now uses `S.amdgpu.mfma_32x32x8_bf16_f32`, vectorized `raw_buffer_load_x4`, LDS staging for A/B, and a 4-wave `2 x 2` warp-grid tile with fused bias + ReLU in the substrate kernel. I also kept the forward path free of torch linear algebra and used cached bf16 weight/bias buffers instead of rebuilding fresh tensors each call. Validation passed with the required command. The round-local eval in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p76/optimization_rounds_no_invariants/round1/debug_eval.jsonl) reports `correctness: true` and candidate runtime `14.8 us` on the MI300X.
- total_tokens: 1506445
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 optimization in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p76/optimization_rounds_no_invariants/round2/output_model_new.py). The kernel now uses a 2-buffer LDS pipeline over the K dimension with a loop unrolled by 2 at 32-wide K steps: - preloads two 16-wide MFMA stages into separate LDS buffers, - overlaps global `raw_buffer_load_x4` for future stages with MFMA on already-materialized LDS fragments, - keeps the existing 4-wave 2x2 warp-grid mapping and natural two-half MFMA consumption intact. Validation passed with the required command. The round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p76/optimization_rounds_no_invariants/round2/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 14.8`.
- total_tokens: 179689
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-3 change in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p76/optimization_rounds_no_invariants/round3/output_model_new.py). The kernel now uses per-thread dynamic `range` values on the full-tensor raw buffer descriptors so the unconditional K-prefetch `raw_buffer_load_x4` calls safely zero-fill past the row boundary, and the explicit in-loop OOB guards were removed. Validation passed with the required command. The round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p76/optimization_rounds_no_invariants/round3/debug_eval.jsonl) reports `compiled: true` and `correctness: true`.
- total_tokens: 339232
