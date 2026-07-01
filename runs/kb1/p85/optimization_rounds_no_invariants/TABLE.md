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
| 1 | 1 | completed | True | True | - | 1065336 | - | - |
| 2 | 2 | completed | True | True | - | 1062081 | - | - |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the round output in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p85/optimization_rounds_no_invariants/round1/output_model_new.py) and updated the callsites in [test_conv2d_asym.py](/workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py) and [bench_conv2d.py](/workspace/substrate/benchmark/conv2d/bench_conv2d.py) to point at this round. The kernel now uses a live substrate MFMA op, keeps the supported depthwise Conv2D behavior, and makes the optimized path graph-safe by caching the device weight tensor until the source storage pointer changes. I also wrote the required case file at [case.txt](/workspace/kernel_benchmark/path/to/case.txt). Validation: `python -m pytest /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py` passed with `2 passed in 3.29s`.
- total_tokens: 1065336
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the round-2 split-K path in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p85/optimization_rounds_no_invariants/round2/output_model_new.py): the main kernel now launches with `SPLIT_K_SLICES = 2`, accumulates per-split fp32 partials into a cached fp32 workspace via `S.amdgpu.buffer_atomic_add_f32`, and a separate finalize kernel converts that workspace back to bf16 NCHW output. The MFMA path is live with `S.amdgpu.mfma_32x32x8_bf16_f32`, and the Python side keeps cached weight/workspace/output buffers so forward does not rebuild device-side metadata tensors. I also updated the current-round callsites in [test_conv2d_asym.py](/workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py) and [bench_conv2d.py](/workspace/substrate/benchmark/conv2d/bench_conv2d.py), and wrote [case.txt](/workspace/kernel_benchmark/path/to/case.txt). Validation: `python -m pytest /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py` passed with `2 passed in 3.30s`.
- total_tokens: 1062081
