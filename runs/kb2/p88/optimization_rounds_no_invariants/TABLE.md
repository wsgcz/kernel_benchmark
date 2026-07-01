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
| 1 | 1 | eval_failed | True | False | - | 2026788 | 0.476 | - |
| 2 | 2 | eval_failed | True | False | - | 1094232 | 0.474 | - |
| 3 | 3 | completed | True | True | 1.0085 | 605599 | 0.474 | 0.470 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: eval_failed
- unlocked hints: 1
- agent: codex
- summary: Implemented a standalone substrate kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p88/optimization_rounds_no_invariants/round1/output_model_new.py) that uses `S.amdgpu.mfma_32x32x8_bf16_f32`, `S.amdgpu.raw_buffer_load_x4`, and LDS staging, and kept `candidate_input.py` unchanged. I also ran the required evaluation command. The kernel compiles and runs, but it does not pass correctness yet: the current failure is an MFMA operand/output layout mismatch, not a compile/runtime failure. The latest round-local result is in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p88/optimization_rounds_no_invariants/round1/debug_eval.jsonl).
- total_tokens: 2026788
- error: Output mismatch
### round2
- status: eval_failed
- unlocked hints: 2
- agent: codex
- summary: Wrote `output_model_new.py` and validated with the exact mandated command. Current status: the inherited baseline kernel from `candidate_input.py` itself fails correctness under the round-local harness with the same mismatch I saw after the pipeline edits (`max_difference=10.5`, `avg_difference=0.208008`). I restored the simpler baseline GEMM path in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p88/optimization_rounds_no_invariants/round2/output_model_new.py) rather than leaving a more complex unverified variant. The latest debug log is in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p88/optimization_rounds_no_invariants/round2/debug_eval.jsonl).
- total_tokens: 1094232
- error: Output mismatch
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Wrote the final implementation to [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p88/optimization_rounds_no_invariants/round3/output_model_new.py). I left `candidate_input.py` unchanged. Validation passed with the required harness command. The round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p88/optimization_rounds_no_invariants/round3/debug_eval.jsonl) shows `correctness: true` and `runtime_us: 0.479` versus `ref_runtime_us: 0.48`.
- total_tokens: 605599
