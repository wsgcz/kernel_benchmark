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
| 1 | 1 | completed | True | True | 0.1747 | 1280724 | 1.310 | 7.500 |
| 2 | 2 | completed | True | True | 0.1212 | 537743 | 0.858 | 7.080 |
| 3 | 3 | completed | True | True | 0.1218 | 303894 | 0.862 | 7.080 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p07/optimization_rounds/round1/output_model_new.py). It now uses `S.amdgpu.mfma_32x32x8_bf16_f32`, stages A/B through LDS, loads global memory with `S.amdgpu.raw_buffer_load_x4`, and scales the MFMA path across a 2x2 warp grid with the fixed accumulator writeback mapping. Validation passed with the required harness command. The round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p07/optimization_rounds/round1/debug_eval.jsonl) shows `correctness: true` and candidate `runtime_us: 7.98`.
- total_tokens: 1280724
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p07/optimization_rounds/round2/output_model_new.py). I left `candidate_input.py` unchanged. The kernel keeps the 4-wave MFMA structure and adds a double-buffered LDS layout with an explicitly unrolled 4-tile K schedule, loading the next tile into the alternate LDS stage before consuming the current one. Validation passed with the required command: correctness `true`, compiled `true`, candidate runtime about `7.08 us` on the recorded run.
- total_tokens: 537743
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round output kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p07/optimization_rounds/round3/output_model_new.py) and kept `candidate_input.py` unchanged. Validation passed with the required command. The round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p07/optimization_rounds/round3/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and `runtime_us: 7.12`.
- total_tokens: 303894
