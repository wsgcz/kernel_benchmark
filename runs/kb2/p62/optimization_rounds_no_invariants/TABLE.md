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
| 1 | 1 | completed | True | True | 0.0746 | 3129798 | 0.573 | 7.680 |
| 2 | 2 | completed | True | True | 0.0736 | 332934 | 0.568 | 7.720 |
| 3 | 3 | completed | True | True | 0.0751 | 614922 | 0.576 | 7.670 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p62/optimization_rounds_no_invariants/round1/output_model_new.py). It now uses a 4-wave AMDGPU MFMA GEMM path with `S.amdgpu.mfma_32x32x8_bf16_f32`, vectorized `raw_buffer_load_x4`, LDS staging for A/B, and a separate substrate post kernel for group norm + leaky ReLU + multiply-by-2. `candidate_input.py` was not modified. Validation passed with the required harness command. The latest round-local result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p62/optimization_rounds_no_invariants/round1/debug_eval.jsonl) shows `correctness: true` and candidate `runtime_us: 7.68`.
- total_tokens: 3129798
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the software-pipelined GEMM update in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p62/optimization_rounds_no_invariants/round2/output_model_new.py). The GEMM kernel now uses a 2-stage LDS ping-pong buffer, preloads two `BLOCK_K=16` tiles, unrolls the K loop by 2, and keeps the per-wave MFMA path unchanged while overlapping the next stage’s global loads with the current pair’s MFMA work. Validation passed with the required command. The round-local result in `debug_eval.jsonl` shows `correctness: true` and candidate `runtime_us: 7.68`.
- total_tokens: 332934
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Wrote the final kernel to [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p62/optimization_rounds_no_invariants/round3/output_model_new.py). The working change is a peeled K-loop in `gemm_bias_mfma_kernel`: the hot loop now runs `k_tiles // 2 - 1` iterations without the per-iteration end check, and the final MFMA pair is handled in a short epilogue. I kept the original correct raw-buffer addressing path after the speculative range-based version produced output mismatches. Validation passed with the required command. The round-local debug result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p62/optimization_rounds_no_invariants/round3/debug_eval.jsonl) shows `correctness: true` and candidate runtime `7.66 us`.
- total_tokens: 614922
