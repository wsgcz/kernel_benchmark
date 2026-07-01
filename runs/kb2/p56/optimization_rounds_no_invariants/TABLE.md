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
| 1 | 1 | completed | True | True | 0.0003 | 1581722 | 0.743 | 2930.000 |
| 2 | 2 | completed | True | True | 0.0003 | 914478 | 0.761 | 2730.000 |
| 3 | 3 | completed | True | True | 0.0003 | 612402 | 0.752 | 2830.000 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized standalone substrate kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p56/optimization_rounds_no_invariants/round1/output_model_new.py). It uses `S.amdgpu.raw_buffer_load_x4`, stages A/B through LDS, issues `S.amdgpu.mfma_32x32x8_bf16_f32` twice per K-step to realize a cooperative `32x32x16` accumulation per wave, and scales that invariant to a 2x2 wave grid by adding only warp ownership offsets for operand fetch and writeback. I also cached the bf16 weight/bias copies in `ModelNew.forward()` so no descriptor-like device metadata is rebuilt per call. Validation passed with the exact required harness command. The recorded result in `debug_eval.jsonl` is `compiled: true`, `correctness: true`, with a measured candidate runtime of about `2930.0 us`.
- total_tokens: 1581722
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the updated kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p56/optimization_rounds_no_invariants/round2/output_model_new.py). The kernel keeps the 4-wave 2x2 warp-grid MFMA structure, stages A/B through LDS with `raw_buffer_load_x4`, and unrolls the K loop by 2 using two LDS stages per iteration. I kept the cudagraph-safe parameter caching path in `ModelNew.forward()`. I was not able to obtain a completed harness result from the required evaluation command. The command repeatedly stayed active without producing output or writing `debug_eval.jsonl`, so I do not have a correctness-pass artifact to report.
- total_tokens: 914478
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Updated [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p56/optimization_rounds_no_invariants/round3/output_model_new.py) only. I kept the existing pipelined MFMA/LDS structure and applied the round’s range-based buffer-op optimization on the scalar bias path by creating a ranged `bias_rsrc` and replacing direct `BIAS0[...]` reads with `S.amdgpu.raw_buffer_load_x1(...)`. I also tried the same approach for the output store, but `raw_buffer_store_x1` is not implemented for `bf16` here, so I reverted that specific part to preserve a working kernel. Validation passed with the required command. The round-local result in `debug_eval.jsonl` reports `compiled: true` and `correctness: true` on `gfx942`, with measured kernel runtime `2830.0 us`.
- total_tokens: 612402
