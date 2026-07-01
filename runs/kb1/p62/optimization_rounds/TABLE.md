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
| 1 | 1 | completed | True | True | - | 1892103 | - | - |
| 2 | 2 | completed | True | True | - | 642494 | - | - |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the MFMA-only conv kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p62/optimization_rounds/round1/output_model_new.py), keeping the fixed `128 x 128` block tile, `2 x 2` warp grid, `32 x 32 x 8` BF16 MFMA fragments, and the required non-contiguous accumulator writeback mapping. The model path is cudagraph-safe on the weight side via `_get_cached_weight`, which reuses the device tensor until the weight storage pointer changes. I also repointed the required callsites in [test_conv2d_asym.py](/workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py) and [bench_conv2d.py](/workspace/substrate/benchmark/conv2d/bench_conv2d.py), fixed the benchmark constructor to pass `(kernel_h, kernel_w)`, and wrote [case.txt](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p62/optimization_rounds/round1/path/to/case.txt). Validation: `python -m pytest /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py` passed with `2 passed`.
- total_tokens: 1892103
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the split-K MFMA Conv2D path in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p62/optimization_rounds/round2/output_model_new.py). The kernel now launches with `SPLIT_K_SLICES = 2`, keeps the existing `128x128` / 4-wave / `mfma_32x32x8_bf16_f32` structure, accumulates per-split partial sums into a shared fp32 workspace via `S.amdgpu.buffer_atomic_add_f32`, then runs a second kernel to write bf16 NCHW output. I also cached the workspace/output tensors and kept the weight cache so the optimized path reuses device allocations across forwards. I updated the round callsites in [test_conv2d_asym.py](/workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py) and [bench_conv2d.py](/workspace/substrate/benchmark/conv2d/bench_conv2d.py) to point at this round, and wrote the required [case.txt](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p62/optimization_rounds/round2/path/to/case.txt). Validation: `python -m pytest /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py` passed with `2 passed`.
- total_tokens: 642494
