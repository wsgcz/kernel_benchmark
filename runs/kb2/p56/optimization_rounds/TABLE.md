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
| 1 | 1 | completed | True | True | 0.0001 | 1378240 | 0.764 | 6220.000 |
| 2 | 2 | completed | True | True | 0.0001 | 1118377 | 0.757 | 6040.000 |
| 3 | 3 | completed | True | True | 0.0001 | 388990 | 0.752 | 6040.000 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented a new MFMA-based kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p56/optimization_rounds/round1/output_model_new.py). Status: the file was written and uses `S.amdgpu.mfma_32x32x8_bf16_f32`, `S.amdgpu.raw_buffer_load_x4`, a 4-wave `2x2` warp grid, and LDS staging. I was not able to get this round to a correctness-passing harness result within the available time. The earlier harness run produced a lowering failure around the MFMA operand construction; the follow-up packed-LDS revision then entered a long compile/run path and did not complete during validation.
- total_tokens: 1378240
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the pipelined MFMA kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p56/optimization_rounds/round2/output_model_new.py). The kernel now uses double-buffered LDS staging for A/B, split LDS consumption into low/high BF16 halves feeding `S.amdgpu.mfma_32x32x8_bf16_f32`, and runs the K loop as a 2-tile software-pipelined steady state with a prologue/epilogue. Validation passed with the required harness command. The latest debug eval in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb2/p56/optimization_rounds/round2/debug_eval.jsonl) reports `compiled: true`, `correctness: true`, and `runtime_us: 6040.0`.
- total_tokens: 1118377
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: `output_model_new.py` now contains the final kernel at [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb2/p56/optimization_rounds/round3/output_model_new.py). I left the kernel logic effectively unchanged because this round’s starting kernel already uses ranged AMD buffer resources for the `raw_buffer_load_x4` path via `S.amdgpu.make_rsrc(..., sixteen_i32)`, and there were no explicit OOB guard branches in the kernel body to remove safely. I verified the file parses with `py_compile`. I also launched the required round-local harness command, but I did not get a correctness result: `/workspace/kb_eval_pipeline/runs/kb2/p56/optimization_rounds/round3/debug_eval.jsonl` stayed empty while the process contended with other long-running GPU eval jobs on the machine.
- total_tokens: 388990
