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
| 1 | 1 | completed | True | True | - | 516336 | - | - |
| 2 | 2 | completed | True | True | - | 2486008 | - | - |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the MFMA-based depthwise conv kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p85/optimization_rounds/round1/output_model_new.py), preserving the fixed `128x128` block tile, fixed `2x2` warp grid, fixed `32x32x8` BF16 MFMA fragments, and the required accumulator writeback mapping. The optimized path is cudagraph-safe for weight metadata/tensors via `_get_cached_weight`, with no convolution fallback in `forward()`. I also updated the listed callsites to this round: [test_conv2d_asym.py](/workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py) [bench_conv2d.py](/workspace/substrate/benchmark/conv2d/bench_conv2d.py) The required case marker was written at [case.txt](/workspace/kb_eval_pipeline/runs/kb1/p85/optimization_rounds/round1/path/to/case.txt), and `python -m pytest /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py` passed with `2 passed`.
- total_tokens: 516336
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the split-K round2 kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p85/optimization_rounds/round2/output_model_new.py). It keeps the `128 x 128` MFMA tile and fixed lane/accumulator mapping, launches `SPLIT_K_SLICES = 2`, atomically reduces per-split fp32 partials into an fp32 workspace, then runs a separate finalize kernel to write bf16 NCHW output. I also kept weight caching and added cached workspace/output buffers for the optimized path. I updated the round selectors in [test_conv2d_asym.py](/workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py) and [bench_conv2d.py](/workspace/substrate/benchmark/conv2d/bench_conv2d.py), and wrote [case.txt](/workspace/kernel_benchmark/path/to/case.txt). Validation: `python -m pytest /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py` passed with `2 passed`.
- total_tokens: 2486008
