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
| 1 | 1 | completed | True | True | 0.0184 | 1330522 | 0.546 | 29.700 |
| 2 | 2 | completed | True | True | 0.0172 | 736389 | 0.545 | 31.700 |
| 3 | 3 | completed | True | True | 0.0169 | 664660 | 0.535 | 31.700 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p53/optimization_rounds/round1/output_model_new.py). It now uses a 4-wave `2 x 2` warp grid, stages A/B through LDS from `raw_buffer_load_x4`, assembles the verified per-lane BF16 fragments, issues `S.amdgpu.mfma_32x32x8_bf16_f32` twice per K-step for a cooperative `32x32x16`, and keeps the fused bias/scale/hardtanh/GELU epilogue in the substrate kernel. Validation passed with the required command. The round-local eval reported `correctness: true` and candidate runtime `29.5 us` in `debug_eval.jsonl`.
- total_tokens: 1330522
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p53/optimization_rounds/round2/output_model_new.py). The kernel now keeps the MFMA-based 4-wave `2 x 2` warp tiling, stages A/B through double-buffered LDS tiles, unrolls the K-tile loop by 2, and interleaves next-tile `raw_buffer_load_x4` fetches with fragment staging and MFMA issue. Validation passed with the required command. The round-local debug result shows `correctness: true` and `runtime_us: 31.6` in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p53/optimization_rounds/round2/debug_eval.jsonl).
- total_tokens: 736389
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Wrote the final kernel to [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p53/optimization_rounds/round3/output_model_new.py) without modifying `candidate_input.py`. Validation passed with the required command. The round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p53/optimization_rounds/round3/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and candidate runtime `31.6 us`.
- total_tokens: 664660
