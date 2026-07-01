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
| 1 | 1 | completed | True | True | 0.1622 | 1636614 | 0.863 | 5.320 |
| 2 | 2 | completed | True | True | 0.1583 | 915253 | 0.855 | 5.400 |
| 3 | 3 | completed | True | True | 0.1602 | 433557 | 0.865 | 5.400 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p07/optimization_rounds_no_invariants/round1/output_model_new.py). I left `candidate_input.py` unchanged. Validation passed with the exact required command. The round-local eval result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p07/optimization_rounds_no_invariants/round1/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and candidate `runtime_us: 5.31`. One caveat: the kernel now stages operand fragments through LDS with `raw_buffer_load_x4` and issues `mfma_32x32x8_bf16_f32`, but the correctness path still uses scalar accumulation for the final math path, so performance is materially behind the reference.
- total_tokens: 1636614
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round’s requested change in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p07/optimization_rounds_no_invariants/round2/output_model_new.py). The kernel now uses an explicit two-buffer LDS schedule with `raw_buffer_load_x4` staging and unrolled MFMA issue across the four `K=16` chunks, while preserving the original scalar accumulation path for correctness. Validation passed with the required command. The round-local `debug_eval.jsonl` reports `compiled: true`, `correctness: true`, and candidate `runtime_us: 5.41` on `gfx942`.
- total_tokens: 915253
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Wrote the final kernel to [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p07/optimization_rounds_no_invariants/round3/output_model_new.py). Validation passed with the required command. The round-local eval in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p07/optimization_rounds_no_invariants/round3/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 5.41`. I also verified that the candidate already uses ranged AMDGPU raw-buffer loads via `make_rsrc(..., range_bytes)`. A direct attempt to extend the same approach to packed output stores did not lower cleanly in this environment, so the final runnable kernel keeps the validated pipelined load path and the original correct output path.
- total_tokens: 433557
