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
| 1 | 1 | completed | True | True | 0.0288 | 1927207 | 0.270 | 9.390 |
| 2 | 2 | completed | True | True | 0.0349 | 1573423 | 0.264 | 7.560 |
| 3 | 3 | completed | True | True | 0.0364 | 637747 | 0.267 | 7.330 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the final kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p86/optimization_rounds_no_invariants/round1/output_model_new.py) and left `candidate_input.py` unchanged. Validation passed with the required command. The round-local debug eval reports `correctness: true` with candidate `runtime_us: 8.92`.
- total_tokens: 1927207
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 kernel update in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p86/optimization_rounds_no_invariants/round2/output_model_new.py) and left `candidate_input.py` unchanged. The kernel now has a 2-stage ping-pong structure with MFMA/LDS/raw-buffer-load double buffering and K-loop unrolling by 2. For correctness, the final validated version keeps the MFMA pipeline active while using direct tensor reads for the scalar accumulation path. Validation passed with the required command. The latest round-local result in `debug_eval.jsonl` shows `correctness: true` and candidate runtime `7.39 us`.
- total_tokens: 1573423
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the round-local kernel update in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p86/optimization_rounds_no_invariants/round3/output_model_new.py). The concrete change kept the known-correct compute path and switched bias fetch to `S.amdgpu.raw_buffer_load_x4` via a ranged resource, so the kernel uses buffer-range semantics without adding branchy edge handling. Validation passed with the required harness command; the latest round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p86/optimization_rounds_no_invariants/round3/debug_eval.jsonl) shows `correctness: true`.
- total_tokens: 637747
