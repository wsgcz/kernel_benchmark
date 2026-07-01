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
| 1 | 1 | completed | True | True | 1.6062 | 761104 | 0.877 | 0.546 |
| 2 | 2 | completed | True | True | 0.0358 | 882727 | 0.551 | 15.400 |
| 3 | 3 | completed | True | True | 0.0345 | 262336 | 0.542 | 15.700 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized model in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p08/optimization_rounds_no_invariants/round1/output_model_new.py). I left `candidate_input.py` unchanged. Validation passed with the exact required round-local command. The generated [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p08/optimization_rounds_no_invariants/round1/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and a measured `runtime_us` of `0.55`.
- total_tokens: 761104
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p08/optimization_rounds_no_invariants/round2/output_model_new.py). It now uses a standalone substrate MFMA path with: - `S.amdgpu.mfma_32x32x8_bf16_f32` - vectorized `raw_buffer_load_x4` - LDS staging for A/B - 4-wave `2 x 2` wave grid over a `64 x 64` block - double-buffered K-tiling with the K-loop unrolled by 2 - host-side prepacking into the MFMA-native 4-column-interleaved fragment layout - cached device buffers so `forward()` does not build descriptor-like device metadata each call Validation passed with the required command. The round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p08/optimization_rounds_no_invariants/round2/debug_eval.jsonl) reports `compiled: true` and `correctness: true`.
- total_tokens: 882727
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p08/optimization_rounds_no_invariants/round3/output_model_new.py). The change removes the host-side `M/N` input padding path by switching `A_pack` and `B_pack` to exact `(M, ...)` and `(N, ...)` shapes and relying on `S.amdgpu.make_rsrc(..., range_bytes)` plus `raw_buffer_load_x4` range semantics for OOB-zeroed tile loads. The existing software-pipelined MFMA/LDS structure was preserved. Output storage stays padded because scalar `raw_buffer_store_x1` is not valid for `bf16` in this environment. Validation passed with the required command. The round-local debug eval reports `compiled: true`, `correctness: true`, and `runtime_us: 15.6`.
- total_tokens: 262336
