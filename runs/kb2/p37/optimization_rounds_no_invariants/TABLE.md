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
| 1 | 1 | eval_failed | True | False | - | 2472974 | 2.600 | - |
| 2 | 2 | eval_failed | True | False | - | 863115 | 2.960 | - |
| 3 | 3 | completed | True | True | 1.0195 | 930081 | 2.610 | 2.560 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: eval_failed
- unlocked hints: 1
- agent: codex
- summary: Implemented an MFMA-based Substrate path in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p37/optimization_rounds_no_invariants/round1/output_model_new.py) with: - `S.amdgpu.mfma_32x32x8_bf16_f32` - `S.amdgpu.raw_buffer_load_x4` - LDS staging - a 4-wave `2 x 2` wave grid for the GEMM tile - a separate Substrate group-norm pass I ran the required harness command and wrote results to [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p37/optimization_rounds_no_invariants/round1/debug_eval.jsonl). The kernel compiles and runs, but it does not pass correctness yet: `max_difference=0.03125`, `avg_difference=0.001465`, `correctness=false`. I did not modify `candidate_input.py`.
- total_tokens: 2472974
- error: Output mismatch
### round2
- status: eval_failed
- unlocked hints: 2
- agent: codex
- summary: Implemented work in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p37/optimization_rounds_no_invariants/round2/output_model_new.py). I validated with the exact required harness command. The current kernel still does not pass correctness in this round’s harness: `correctness: false`, `max_difference: 0.031250`, `avg_difference: 0.001465` in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p37/optimization_rounds_no_invariants/round2/debug_eval.jsonl). Given the round constraints and the persistent mismatch from the starting baseline, I left `output_model_new.py` in the restored baseline state rather than keeping a less stable pipelined variant.
- total_tokens: 863115
- error: Output mismatch
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Wrote the final model to [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p37/optimization_rounds_no_invariants/round3/output_model_new.py). The working path now matches the round-local reference semantics exactly: `Linear -> torch.sigmoid(x) * x -> + bias -> GroupNorm`. Validation passed with the required command. The latest round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p37/optimization_rounds_no_invariants/round3/debug_eval.jsonl) reports `correctness: true` with candidate runtime `2.93 us` and reference runtime `2.64 us`.
- total_tokens: 930081
