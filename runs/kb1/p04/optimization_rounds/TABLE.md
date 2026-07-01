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
| 1 | 1 | completed | True | True | 0.0140 | 1812964 | 1.170 | 83.500 |
| 2 | 2 | completed | True | True | 0.0246 | 2137539 | 1.170 | 47.500 |
| 3 | 3 | completed | True | True | 0.0242 | 993602 | 1.150 | 47.500 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p04/optimization_rounds/round1/output_model_new.py). Validation passed with the required round-local command: `compiled: true`, `correctness: true`. The measured candidate runtime was `80.0 us` on `gfx942`. One limitation remains: the final working version uses MFMA and LDS staging with the required accumulator unpacking, but the `raw_buffer_load_x4` path hit substrate codegen/runtime constraints for this case, so the validated kernel falls back to direct tensor indexing for operand assembly before MFMA.
- total_tokens: 1812964
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p04/optimization_rounds/round2/output_model_new.py). The kernel now keeps the MFMA path, uses two-slot LDS double buffering, and unrolls the K loop by 2 with overlapped preload/compute scheduling. `B` is fetched via `S.amdgpu.raw_buffer_load_x4`, while `A` stays on direct row loads to preserve correctness under the current Substrate descriptor constraints. Validation passed with the required command. The latest round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p04/optimization_rounds/round2/debug_eval.jsonl) shows `correctness: true` and candidate `runtime_us: 48.0`.
- total_tokens: 2137539
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Wrote the final kernel to [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p04/optimization_rounds/round3/output_model_new.py). Validation passed with the required command. The recorded result in `debug_eval.jsonl` shows `compiled: true` and `correctness: true`, with candidate runtime `47.5 us`. I also tested a range-based raw buffer store rewrite for the final writeback, but Substrate lowering blocked the required `bf16` packing path. The final kernel therefore keeps the existing pipelined MFMA structure and the range-based `raw_buffer_load_x4` path that already lowers and runs correctly.
- total_tokens: 993602
