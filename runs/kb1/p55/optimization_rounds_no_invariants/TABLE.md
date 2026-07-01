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
| 1 | 1 | completed | True | True | - | 3297357 | - | - |
| 2 | 2 | completed | True | True | - | 285721 | - | - |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the MFMA Conv2D kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p55/optimization_rounds_no_invariants/round1/output_model_new.py), updated the two required callsites, and wrote `/workspace/kernel_benchmark/path/to/case.txt` with the round directory path. Validation passed: ```text python -m pytest /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py 2 passed in 6.99s ``` The optimized path uses `S.amdgpu.mfma_32x32x8_bf16_f32`, keeps the existing supported shape/behavior, and caches prepared device weights so `forward()` does not rebuild transient metadata-style state every call.
- total_tokens: 3297357
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the split-K MFMA Conv2D path in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p55/optimization_rounds_no_invariants/round2/output_model_new.py). The new kernel keeps the fixed `128x128` 4-wave MFMA tiling, extends the launch grid by `SPLIT_K_SLICES = 2`, accumulates per-split fp32 partials into a shared fp32 workspace with `S.amdgpu.buffer_atomic_add_f32`, and finalizes into bf16 NCHW output with a second kernel. The forward path is cudagraph-safe in the requested sense: it does not build descriptor or metadata device tensors in `forward()`, and it caches the bf16 weight conversion plus reusable workspace. I also updated the allowed callsites to import `round2` in [test_conv2d_asym.py](/workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py) and [bench_conv2d.py](/workspace/substrate/benchmark/conv2d/bench_conv2d.py), and wrote [case.txt](/workspace/kb_eval_pipeline/runs/kb1/p55/optimization_rounds_no_invariants/round2/path/to/case.txt). Validation: `python -m pytest /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py` passed with `2 passed in 7.26s`.
- total_tokens: 285721
