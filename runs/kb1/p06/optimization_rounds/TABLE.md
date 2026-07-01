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
| 1 | 1 | completed | True | True | 0.0016 | 392053 | 0.527 | 331.000 |
| 2 | 2 | completed | True | True | 0.0016 | 234570 | 0.527 | 331.000 |
| 3 | 3 | completed | True | True | 0.0016 | 559124 | 0.522 | 327.000 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the MFMA-based kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p06/optimization_rounds/round1/output_model_new.py). `candidate_input.py` was not modified. Validation passed with the exact round-local command. The recorded result in `debug_eval.jsonl` shows `compiled: true`, `correctness: true`, and `runtime_us: 330.0`.
- total_tokens: 392053
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the pipelined MFMA kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p06/optimization_rounds/round2/output_model_new.py). The K-loop is now double-buffered through LDS, unrolled by 2 tiles, and scheduled so tile 0 compute, next tile 0 refill, tile 1 compute, next tile 1 refill happen in a fixed steady-state pattern without changing the MFMA lane/output mapping. Validation passed with the required command. The round-local harness recorded `compiled: true`, `correctness: true`, and `runtime_us: 327.0` in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p06/optimization_rounds/round2/debug_eval.jsonl).
- total_tokens: 234570
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the final kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p06/optimization_rounds/round3/output_model_new.py) and left `candidate_input.py` unchanged. Validation passed with the required command. `debug_eval.jsonl` reports `compiled: true`, `correctness: true`, and candidate runtime `329.0 us` on `gfx942`. The environment rejected the unsupported 5-argument `raw_buffer_load_x4` form, so the final kernel keeps the working pipelined MFMA kernel with descriptor-range-backed raw buffer loads via `make_rsrc`, which is the supported path in this workspace.
- total_tokens: 559124
