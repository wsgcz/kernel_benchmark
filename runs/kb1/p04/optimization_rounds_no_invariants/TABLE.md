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
| 1 | 1 | completed | True | True | 0.0081 | 2029245 | 1.700 | 210.000 |
| 2 | 2 | completed | True | True | 0.0038 | 276727 | 1.170 | 309.000 |
| 3 | 3 | completed | True | True | 0.0038 | 916650 | 1.160 | 309.000 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p04/optimization_rounds_no_invariants/round1/output_model_new.py). The original [candidate_input.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p04/optimization_rounds_no_invariants/round1/candidate_input.py) was left unchanged. Validation passed with the exact required command. The round-local debug result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p04/optimization_rounds_no_invariants/round1/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and candidate `runtime_us: 208.0`.
- total_tokens: 2029245
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 optimization in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p04/optimization_rounds_no_invariants/round2/output_model_new.py). The kernel now uses a 2-stage LDS double buffer, unrolls the K-loop by 2, and interleaves `raw_buffer_load_x4`, LDS staging, and the two natural `mfma_32x32x8_bf16_f32` steps per `K_TILE` while preserving the 2x2 wave-grid mapping and existing cudagraph-safe B-cache behavior. Validation passed with the required command. `debug_eval.jsonl` reports `compiled: true`, `correctness: true`, and `runtime_us: 312.0`.
- total_tokens: 276727
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-3 change in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p04/optimization_rounds_no_invariants/round3/output_model_new.py): the explicit `row < M` output-store guards were removed and replaced with `S.amdgpu.raw_buffer_store_x2(...)` using descriptor range bytes so OOB writes are discarded by the hardware path. The existing pipelined MFMA/LDS/global-load structure was left intact. Validation passed with the required command. The round-local eval in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p04/optimization_rounds_no_invariants/round3/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 310.0`.
- total_tokens: 916650
