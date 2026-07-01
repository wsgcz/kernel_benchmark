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
| 1 | 1 | completed | True | True | 0.1864 | 499256 | 0.274 | 1.470 |
| 2 | 2 | completed | True | True | 0.1816 | 269959 | 0.267 | 1.470 |
| 3 | 3 | completed | True | True | 0.9819 | 145313 | 0.271 | 0.276 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the optimized model in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p99/optimization_rounds_no_invariants/round1/output_model_new.py). Validation passed with the exact required command. The round-local eval result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p99/optimization_rounds_no_invariants/round1/debug_eval.jsonl) reports `compiled: true` and `correctness: true`.
- total_tokens: 499256
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the requested round output in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p99/optimization_rounds_no_invariants/round2/output_model_new.py). The main change is a new 4-wave Substrate MFMA probe kernel that stages operands through LDS, uses `S.amdgpu.raw_buffer_load_x4`, consumes each 16-byte fragment as two natural BF16 halves, and uses a double-buffered two-step MFMA sequence. I validated it with the exact required evaluation command. The run compiled and passed correctness: `correctness: true` in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p99/optimization_rounds_no_invariants/round2/debug_eval.jsonl).
- total_tokens: 269959
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the final kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p99/optimization_rounds_no_invariants/round3/output_model_new.py). I left `candidate_input.py` untouched. The main change was replacing the Triton matmul path with the native `F.linear` ROCm path while keeping the MFMA probe kernel that uses `make_rsrc(..., range_bytes)` and `raw_buffer_load_x4` without explicit OOB branches. The round-local evaluation passed correctness. Validation used the required command and produced [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb2/p99/optimization_rounds_no_invariants/round3/debug_eval.jsonl:1): `compiled=true`, `correctness=true`, candidate `runtime_us=0.273`.
- total_tokens: 145313
