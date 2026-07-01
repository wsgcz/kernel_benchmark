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
| 1 | 1 | completed | True | True | 0.0031 | 1448862 | 0.528 | 171.000 |
| 2 | 2 | completed | True | True | 0.0037 | 614072 | 0.533 | 143.000 |
| 3 | 3 | completed | True | True | 0.0037 | 324174 | 0.536 | 144.000 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the final kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p06/optimization_rounds_no_invariants/round1/output_model_new.py). It uses `S.amdgpu.mfma_32x32x8_bf16_f32` on a 4-wave `2x2` wave grid, with `S.amdgpu.raw_buffer_load_x4` feeding packed MFMA fragments and a shape-specific launch over the full `256x256` output. I ran the exact required evaluation command and it passed correctness. The round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p06/optimization_rounds_no_invariants/round1/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 169.0`.
- total_tokens: 1448862
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p06/optimization_rounds_no_invariants/round2/output_model_new.py). The kernel now stages the two unique A tiles and two unique B tiles through two LDS buffers, keeps the four-wave `2 x 2` warp grid, and runs the K loop as an unrolled-by-2 software pipeline with `compute even -> refill even -> compute odd -> refill odd` around `mfma_32x32x8_bf16_f32`. Validation passed with the required harness command. The debug eval reported `compiled: true`, `correctness: true`, and candidate `runtime_us: 146.0`.
- total_tokens: 614072
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round update in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p06/optimization_rounds_no_invariants/round3/output_model_new.py). The kernel keeps the existing double-buffered pipelined MFMA structure and now uses explicit range-backed raw buffer resources for the packed A/B loads; I left the final C write as tensor indexing because `raw_buffer_store_x1` in this environment rejects `bf16` and only accepts `i32`. Validation passed with the required command. The recorded result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p06/optimization_rounds_no_invariants/round3/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and `runtime_us: 145.0`. `candidate_input.py` was not modified.
- total_tokens: 324174
