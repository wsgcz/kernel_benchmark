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
| 1 | 1 | completed | True | True | - | 912458 | - | - |
| 2 | 2 | completed | True | True | - | 819722 | - | - |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the MFMA-based round1 kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p56/optimization_rounds/round1/output_model_new.py), with the fixed `128 x 128` block tile, fixed `2 x 2` wave grid, required fragment packing, and the exact accumulator unpack/writeback mapping from the prompt. The optimized path caches the packed device weight outside `forward()` and rebuilds it only when the underlying weight storage pointer changes. I also updated the round-specific callsites in [test_conv2d_asym.py](/workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py) and [bench_conv2d.py](/workspace/substrate/benchmark/conv2d/bench_conv2d.py), and wrote `/workspace/kernel_benchmark/path/to/case.txt` with the current round directory. Validation: `python -m pytest /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py` passed (`2 passed`).
- total_tokens: 912458
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the split-K round in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p56/optimization_rounds/round2/output_model_new.py). The kernel now keeps the existing `128 x 128` / 4-wave MFMA structure, splits K across `SPLIT_K_SLICES = 2`, accumulates partial fp32 sums into a GEMM-major fp32 workspace with `S.amdgpu.buffer_atomic_add_f32`, and runs a second kernel to remap that workspace back to bf16 NCHW output. The optimized path also caches the device weight tensor plus reusable workspace/output buffers so `forward()` does not rebuild device metadata/tensors each call. I also updated the round-specific callsites in [test_conv2d_asym.py](/workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py) and [bench_conv2d.py](/workspace/substrate/benchmark/conv2d/bench_conv2d.py), and wrote the required marker file at [case.txt](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p56/optimization_rounds/round2/path/to/case.txt). Validation: `python -m pytest /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py` passed with `2 passed`.
- total_tokens: 819722
