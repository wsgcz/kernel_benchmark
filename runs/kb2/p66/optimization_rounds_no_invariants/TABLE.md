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
| 1 | 1 | completed | True | True | 0.0275 | 1157916 | 0.263 | 9.550 |
| 2 | 2 | completed | True | True | 0.0346 | 620764 | 0.259 | 7.480 |
| 3 | 3 | completed | True | True | 0.0348 | 664279 | 0.261 | 7.490 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p66/optimization_rounds_no_invariants/round1/output_model_new.py). `candidate_input.py` was not modified. Validation passed with the required round-local harness: correctness `true`, compiled `true`, and measured candidate runtime `9.55 us` on `gfx942`.
- total_tokens: 1157916
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 optimization in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p66/optimization_rounds_no_invariants/round2/output_model_new.py). The GEMM kernel now uses a double-buffered LDS staging path with a K-loop unrolled by 2, explicit four-wave `2 x 2` ownership, and MFMA issued from LDS-reloaded BF16 fragments while interleaving reloads with compute. Validation passed with the required command. The round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p66/optimization_rounds_no_invariants/round2/debug_eval.jsonl) reports `compiled: true` and `correctness: true`.
- total_tokens: 620764
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round change in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p66/optimization_rounds_no_invariants/round3/output_model_new.py). The kernel now removes the explicit output-row guard and keeps the output path branch-free, while continuing to use ranged `S.amdgpu.raw_buffer_load_x4` descriptors for the buffer-load side. I attempted a matching `raw_buffer_store_*` packing path, but Substrate hit an MLIR-to-LLVM lowering blocker on the required bf16-to-packed-u32 repack, so I kept the store as direct tensor writes to preserve a passing executable kernel. Validation passed with the required command. `debug_eval.jsonl` reports `compiled: true`, `correctness: true`, and candidate `runtime_us: 7.49`.
- total_tokens: 664279
