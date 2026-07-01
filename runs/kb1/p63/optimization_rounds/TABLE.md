# Conv2D Optimization History

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
| 1 | 1 | eval_failed | False | False | - | 1017690 | - | - |
| 2 | 2 | completed | True | True | - | 486145 | - | - |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: eval_failed
- unlocked hints: 1
- agent: codex
- summary: Implemented the MFMA conv2d kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p63/optimization_rounds/round1/output_model_new.py), kept the `128x128` block tile and fixed `2x2` warp-grid / `32x32x8` BF16 MFMA mapping, and added graph-safe weight caching via `_get_cached_weight` so `forward()` does not rebuild device-side cached state each call. Updated the round path references in [test_conv2d_asym.py](/workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py) and [bench_conv2d.py](/workspace/substrate/benchmark/conv2d/bench_conv2d.py), and wrote [case.txt](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p63/optimization_rounds/round1/case.txt). Validation: `python -m pytest /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py` Result: `2 passed in 1.83s`
- total_tokens: 1017690
- error: Conv pytest validation failed.
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the split-K MFMA conv kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p63/optimization_rounds/round2/output_model_new.py), keeping the existing `128x128` tile and MFMA lane/writeback mapping, and added a separate finalize kernel that converts the fp32 reduction workspace back to bf16 NCHW output. The forward path now reuses cached weight, workspace, and output tensors so it does not rebuild device-side state each call. I also repointed the round-specific callsites in [test_conv2d_asym.py](/workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py) and [bench_conv2d.py](/workspace/substrate/benchmark/conv2d/bench_conv2d.py), and wrote [case.txt](/workspace/kernel_benchmark/path/to/case.txt). Validation: `python -m pytest /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py` passed with `2 passed`.
- total_tokens: 486145
