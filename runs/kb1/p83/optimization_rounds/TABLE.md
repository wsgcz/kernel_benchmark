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
| 1 | 1 | completed | True | True | - | 1389033 | - | - |
| 2 | 2 | completed | True | True | - | 707833 | - | - |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the round changes in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p83/optimization_rounds/round1/output_model_new.py), updated the callsites in [test_conv2d_asym.py](/workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py) and [bench_conv2d.py](/workspace/substrate/benchmark/conv2d/bench_conv2d.py), and wrote [case.txt](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p83/optimization_rounds/round1/path/to/case.txt). Validation passed with: ```bash python -m pytest /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py ``` Result: `2 passed in 3.25s`
- total_tokens: 1389033
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the Split-K round in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p83/optimization_rounds/round2/output_model_new.py). The kernel now uses a split-K MFMA accumulation pass with `SPLIT_K_SLICES = 2`, atomically reduces partial fp32 sums into a cached fp32 workspace, and runs a separate finalize kernel to write bf16 NCHW output. `forward()` reuses cached weight/workspace/output tensors, so it does not build descriptor or metadata device tensors there. I also updated the local callsites to this round in [test_conv2d_asym.py](/workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py) and [bench_conv2d.py](/workspace/substrate/benchmark/conv2d/bench_conv2d.py), and wrote `/workspace/kernel_benchmark/path/to/case.txt` with the round directory path. Validation: `python -m pytest /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py` passed (`2 passed`).
- total_tokens: 707833
