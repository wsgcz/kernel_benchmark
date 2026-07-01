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
| 1 | 1 | completed | True | True | 0.0102 | 997763 | 0.252 | 24.600 |
| 2 | 2 | completed | True | True | 0.0174 | 2731330 | 0.254 | 14.600 |
| 3 | 3 | completed | True | True | 0.0298 | 266763 | 0.252 | 8.470 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p16/optimization_rounds_no_invariants/round1/output_model_new.py). It now uses `raw_buffer_load_x4`, stages A/B through LDS, issues `S.amdgpu.mfma_32x32x8_bf16_f32` in a 2x2 wave grid, and removes the disallowed `torch.matmul` fallback. Validation passed with the required command. The round-local debug eval reports `correctness: true` and candidate `runtime_us: 24.9` in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p16/optimization_rounds_no_invariants/round1/debug_eval.jsonl).
- total_tokens: 997763
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round update in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p16/optimization_rounds_no_invariants/round2/output_model_new.py). The kernel now uses a double-buffered, unrolled-by-2 K loop with staged LDS loads and pipelined MFMA issue inside the loop, while preserving the known-correct accumulation/writeback path. I validated it with the exact required command. The latest round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p16/optimization_rounds_no_invariants/round2/debug_eval.jsonl) is `correctness: true` with measured candidate runtime `14.5 us`.
- total_tokens: 2731330
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round’s kernel update in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p16/optimization_rounds_no_invariants/round3/output_model_new.py) without touching `candidate_input.py`. The change uses row-subview buffer descriptors with explicit `range_bytes` for A/B raw buffer loads and removes the loop-time OOB guard branches around the double-buffer prefetch path, keeping the unrolled-by-2 K loop structure intact. Validation passed with the required harness command. The recorded result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p16/optimization_rounds_no_invariants/round3/debug_eval.jsonl) shows `compiled: true`, `correctness: true`, and `runtime_us: 8.05`.
- total_tokens: 266763
