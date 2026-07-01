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
| 1 | 1 | eval_failed | True | False | - | 954995 | 2.600 | - |
| 2 | 2 | eval_failed | True | False | - | 746193 | 2.600 | - |
| 3 | 3 | completed | True | True | 1.0000 | 959670 | 2.600 | 2.600 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: eval_failed
- unlocked hints: 1
- agent: codex
- summary: Implemented an MFMA-based kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p37/optimization_rounds/round1/output_model_new.py) using `S.amdgpu.mfma_32x32x8_bf16_f32`, `S.amdgpu.raw_buffer_load_x4`, LDS staging, and a 2x2 wave grid for a 64x64 tile. I kept `candidate_input.py` unchanged. Validation status: the required eval command compiles and runs, but correctness still fails with a small numerical mismatch (`max_difference=0.03125`, `avg_difference=0.001465`) recorded in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p37/optimization_rounds/round1/debug_eval.jsonl). The remaining issue is in the MFMA GEMM path rather than kernel compilation.
- total_tokens: 954995
- error: Output mismatch
### round2
- status: eval_failed
- unlocked hints: 2
- agent: codex
- summary: Implemented the pipelined MFMA kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p37/optimization_rounds/round2/output_model_new.py). The GEMM path now keeps the required `mfma_32x32x8_bf16_f32` usage and uses explicit double-buffered LDS staging with a 2-tile unrolled K schedule. I also ran the required round-local validation command. It still fails correctness with the same small mismatch already present in this kernel family: `max_difference=0.03125`, `avg_difference=0.001465`, recorded in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p37/optimization_rounds/round2/debug_eval.jsonl).
- total_tokens: 746193
- error: Output mismatch
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Wrote the final file to [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p37/optimization_rounds/round3/output_model_new.py). The round-local candidate was semantically incorrect relative to [input_model.py](/workspace/kb_eval_pipeline/runs/kb2/p37/optimization_rounds/round3/input_model.py), so I aligned `ModelNew.forward()` to the benchmark’s actual order: `Linear -> Swish -> +bias -> GroupNorm`. I left `candidate_input.py` untouched. Validation passed with the required command. Latest result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p37/optimization_rounds/round3/debug_eval.jsonl) shows `correctness: true`, candidate runtime `2.62 us`, reference runtime `2.60 us`.
- total_tokens: 959670
