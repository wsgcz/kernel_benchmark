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
| 1 | 1 | completed | True | True | 0.0451 | 575977 | 0.335 | 7.420 |
| 2 | 2 | completed | True | True | 0.0450 | 221881 | 0.334 | 7.430 |
| 3 | 3 | completed | True | True | 0.0449 | 171488 | 0.336 | 7.480 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p95/optimization_rounds_no_invariants/round1/output_model_new.py). It now uses a four-wave `64x64x16` tiled MFMA path with `S.amdgpu.mfma_32x32x8_bf16_f32`, vectorized global loads via `S.amdgpu.raw_buffer_load_x4`, LDS staging for both operands, and per-wave output offsets for the `2 x 2` wave grid. The forward path keeps weights/bias/add tensors cached by storage pointer so it stays cudagraph-safe without rebuilding device-side metadata each call. Validation passed with the required round-local command. The harness result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p95/optimization_rounds_no_invariants/round1/debug_eval.jsonl) shows `compiled: true` and `correctness: true`.
- total_tokens: 575977
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 kernel update in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p95/optimization_rounds_no_invariants/round2/output_model_new.py). The kernel keeps the 4-wave `2 x 2` warp grid and MFMA path intact, adds two-stage LDS buffering, and restructures the K traversal into `32`-wide macro-iterations with two `16`-wide MFMA steps. Validation passed with the exact required command. The recorded result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p95/optimization_rounds_no_invariants/round2/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and `runtime_us: 7.43`.
- total_tokens: 221881
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-3 kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p95/optimization_rounds_no_invariants/round3/output_model_new.py). The change keeps the existing double-buffered MFMA pipeline but removes the K-loop OOB guard branches around the pipelined `raw_buffer_load_x4` stages, relying on the resource `range` behavior instead. Validation passed with the required command. The round-local eval record in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p95/optimization_rounds_no_invariants/round3/debug_eval.jsonl) shows `compiled: true` and `correctness: true`.
- total_tokens: 171488
