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
| 1 | 1 | completed | True | True | 0.0015 | 460696 | 0.334 | 223.000 |
| 2 | 2 | completed | True | True | 0.0015 | 315482 | 0.338 | 230.000 |
| 3 | 3 | completed | True | True | 0.0015 | 276116 | 0.332 | 223.000 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p64/optimization_rounds/round1/output_model_new.py) and left [candidate_input.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p64/optimization_rounds/round1/candidate_input.py) unchanged. Validation passed with the required round-local command. The debug eval in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p64/optimization_rounds/round1/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 227.0`.
- total_tokens: 460696
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 update in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p64/optimization_rounds/round2/output_model_new.py). The linear kernel now uses a 2-stage shared-memory pipeline with double buffering, K-loop unrolling by 2, `raw_buffer_load_x4` for operand fetches, and LDS-backed `mfma_32x32x8_bf16_f32` issue points inside the pipelined loop. `candidate_input.py` was not modified. Validation passed with the exact required command. The round-local eval in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p64/optimization_rounds/round2/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 226.0`.
- total_tokens: 315482
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-3 change in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p64/optimization_rounds/round3/output_model_new.py): the linear kernel now uses ranged AMDGPU raw-buffer resources for `X` and `W`, and the double-buffered K-loop issues unconditional `raw_buffer_load_x4` prefetches for `next_kk0` and `next_kk1`. That removes the explicit end-of-K OOB branches while preserving the existing software-pipelined MFMA/LDS structure. Validation passed with the required command. The round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p64/optimization_rounds/round3/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and `runtime_us: 226.0`.
- total_tokens: 276116
