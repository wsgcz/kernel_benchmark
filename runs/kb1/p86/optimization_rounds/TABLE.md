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
| 1 | 1 | completed | True | True | - | 983272 | - | - |
| 2 | 2 | completed | True | True | - | 1312741 | - | - |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized MFMA-based fused conv kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p86/optimization_rounds/round1/output_model_new.py), keeping the fixed `128x128` block tile, `2x2` warp grid, `64x64` per-warp MFMA work, and the required accumulator unpack/writeback mapping. The runtime path now caches the BF16 depthwise and pointwise weight tensors by storage pointer so `forward()` does not rebuild device-side cached tensors unless the underlying weight storage changes. I also updated the round callsites in [test_conv2d_asym.py](/workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py) and [bench_conv2d.py](/workspace/substrate/benchmark/conv2d/bench_conv2d.py), and wrote [case.txt](/workspace/kb_eval_pipeline/runs/kb1/p86/optimization_rounds/round1/case.txt) with the required single-line round path. `candidate_input.py` was not modified. Verification: `python -m pytest /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py` passed with `2 passed`.
- total_tokens: 983272
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the split-K round in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p86/optimization_rounds/round2/output_model_new.py). The kernel now launches `SPLIT_K_SLICES = 2`, keeps the existing 128x128 / 4-wave MFMA layout, accumulates per-split fp32 partials into a flat fp32 workspace with `S.amdgpu.buffer_atomic_add_f32`, and runs a second kernel to remap that GEMM-major workspace back to bf16 NCHW output. I also kept the weight path cached by storage pointer/device and added cached workspace reuse so `forward()` doesn’t rebuild metadata tensors. I updated the stale conv2d test callsite in [test_conv2d_asym.py](/workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py) to point at this round and to validate the current depthwise/pointwise cache API, and wrote [case.txt](/workspace/kernel_benchmark/path/to/case.txt) with the required round path. The requested validation passed: `python -m pytest /workspace/substrate/test/examples/conv2d/amdgpu/test_conv2d_asym.py` -> `2 passed`. I did not update `/workspace/substrate/benchmark/conv2d/bench_conv2d.py` because that file is not present in this workspace snapshot.
- total_tokens: 1312741
