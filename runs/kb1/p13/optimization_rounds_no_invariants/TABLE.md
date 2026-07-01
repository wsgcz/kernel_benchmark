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
| 1 | 1 | completed | True | True | 0.0241 | 1482960 | 0.226 | 9.380 |
| 2 | 2 | completed | True | True | 0.0179 | 166022 | 0.222 | 12.400 |
| 3 | 3 | completed | True | True | 0.0175 | 621758 | 0.226 | 12.900 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the final kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p13/optimization_rounds_no_invariants/round1/output_model_new.py). It uses `S.amdgpu.mfma_32x32x8_bf16_f32` and `S.amdgpu.raw_buffer_load_x4`, removes the forbidden `torch.matmul` fallback, and keeps the path cudagraph-safe. Validation passed with the required round command. The latest round-local result in `debug_eval.jsonl` shows `correctness: true` and candidate `runtime_us: 9.4`.
- total_tokens: 1482960
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p13/optimization_rounds_no_invariants/round2/output_model_new.py). The kernel now uses double-buffered LDS staging for both the MFMA operand path and the scalar correctness path, with the K loop unrolled by 2 and scheduled as an even/odd software pipeline. Validation passed with the exact required command. The round-local eval result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p13/optimization_rounds_no_invariants/round2/debug_eval.jsonl) shows `compiled: true` and `correctness: true`.
- total_tokens: 166022
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round output in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p13/optimization_rounds_no_invariants/round3/output_model_new.py) without modifying [candidate_input.py](/workspace/kb_eval_pipeline/runs/kb1/p13/optimization_rounds_no_invariants/round3/candidate_input.py). Validation passed with the required command. The round-local eval recorded `correctness: true` in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p13/optimization_rounds_no_invariants/round3/debug_eval.jsonl).
- total_tokens: 621758
