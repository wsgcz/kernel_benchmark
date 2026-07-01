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
| 1 | 1 | completed | True | True | 0.0404 | 493166 | 1.120 | 27.700 |
| 2 | 2 | completed | True | True | 0.0408 | 241016 | 1.130 | 27.700 |
| 3 | 3 | completed | True | True | 0.0403 | 145904 | 1.120 | 27.800 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p40/optimization_rounds_no_invariants/round1/output_model_new.py). It uses a 4-wave `64x64x16` tile, stages operands through LDS, loads global data with `S.amdgpu.raw_buffer_load_x4`, and performs the matmul with two natural-order `S.amdgpu.mfma_32x32x8_bf16_f32` steps per K tile. Validation passed with the required command. The round-local eval in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p40/optimization_rounds_no_invariants/round1/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 27.8`.
- total_tokens: 493166
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the software-pipelined kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p40/optimization_rounds_no_invariants/round2/output_model_new.py). The main change is a double-buffered LDS schedule that keeps the existing 4-wave `2x2` warp tile mapping and MFMA math intact while unrolling the K loop by 2 tiles and interleaving MFMA with staged `raw_buffer_load_x4` refills. Validation passed with the exact required command. The round-local debug eval reports `compiled: true`, `correctness: true`, and `runtime_us: 27.7` on `gfx942`.
- total_tokens: 241016
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-3 change in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p40/optimization_rounds_no_invariants/round3/output_model_new.py): the kernel now relies on the raw-buffer `range` behavior for the speculative next-tile prefetch and removes the explicit `has_next_pair` OOB guard from the pipelined K loop. The double-buffered MFMA/LDS/global overlap structure from the input kernel is preserved. Validation passed with the required command. The round-local harness recorded `compiled: true` and `correctness: true` in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p40/optimization_rounds_no_invariants/round3/debug_eval.jsonl), with measured candidate runtime `28.1 us`.
- total_tokens: 145904
