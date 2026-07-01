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
| 1 | 1 | eval_failed | True | False | - | 1883446 | 0.270 | - |
| 2 | 2 | eval_failed | True | False | - | 1726437 | 0.307 | - |
| 3 | 3 | completed | True | True | 0.9892 | 415893 | 0.274 | 0.277 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: eval_failed
- unlocked hints: 1
- agent: codex
- summary: Implemented a new `output_model_new.py` at [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p14/optimization_rounds/round1/output_model_new.py) with an MFMA probe kernel that uses `S.amdgpu.mfma_32x32x8_bf16_f32`, LDS staging, and `S.amdgpu.raw_buffer_load_x4`, plus a standalone fused substrate path for the benchmark computation. I also ran the required round evaluation command and wrote results to [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p14/optimization_rounds/round1/debug_eval.jsonl). The current result still does **not** pass correctness: the latest harness entry reports `correctness: false` with `max_difference: 32.0` and `avg_difference: 5.3125`.
- total_tokens: 1883446
- error: Output mismatch
### round2
- status: eval_failed
- unlocked hints: 2
- agent: codex
- summary: Wrote the final kernel to [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p14/optimization_rounds/round2/output_model_new.py). I did not get this round to a correctness pass. The closest validated result was a compiled run with output mismatch in `/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p14/optimization_rounds/round2/debug_eval.jsonl`, with `max_difference=8000` on the first MFMA row-reduction attempt, then a best later mismatch of `max_difference=32` / `avg_difference=5.3125` on the safer fused reduction path, but not `allclose`-passing. A subsequent attempt to align numerics with `fp32` weights hit GPU memory faults, so I restored the closest stable variant and stopped.
- total_tokens: 1726437
- error: Output mismatch
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Wrote the final kernel implementation to [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p14/optimization_rounds/round3/output_model_new.py). The forward path now uses a cached transposed weight with the exact original math sequence, which restored correctness while avoiding the prior slow reference kernel path. Validation passed with the required harness command. The latest round-local result in `debug_eval.jsonl` shows `correctness: true` with candidate runtime `0.276` vs reference `0.271` using `cudagraph`.
- total_tokens: 415893
