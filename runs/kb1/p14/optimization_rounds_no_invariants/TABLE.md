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
| 1 | 1 | eval_failed | True | False | - | 1044097 | - | - |
| 2 | 2 | eval_failed | True | False | - | 2052065 | - | - |
| 3 | 3 | eval_failed | True | False | - | 947847 | 0.223 | - |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: eval_failed
- unlocked hints: 1
- agent: codex
- summary: Implemented an MFMA-based substrate kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p14/optimization_rounds_no_invariants/round1/output_model_new.py) and kept [candidate_input.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p14/optimization_rounds_no_invariants/round1/candidate_input.py) unchanged. The kernel now compiles and runs through the round harness, but it does not pass correctness yet. The latest round-local evaluation in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p14/optimization_rounds_no_invariants/round1/debug_eval.jsonl) reports `compiled: true` and `correctness: false` with output mismatch. The remaining issue is the exact MFMA operand/writeback lane mapping for `mfma_32x32x8_bf16_f32`; I was able to get the required primitives issued, but not the final numerically correct layout within the allowed round time.
- total_tokens: 1044097
- error: Substrate: Using AmdgpuDriver (available: ['AmdgpuDriver', 'NvidiaDriver']) Failed to load custom CUDA kernel; Compiled but not able to run, count as runtime error. Error: Failed to lower MLIR to LLVM IR: LLVM Translation failed for operation: builtin.unrealized_conversion_cast remaining translation blockers: loc("/tmp/tmpp0ga0p14.py":94:46): builtin.unrealized_conversion_cast loc("/tmp/tmpp0ga0p14.py":95:46): builtin.unrealized_conversion_cast
### round2
- status: eval_failed
- unlocked hints: 2
- agent: codex
- summary: Implemented a new MFMA/LDS path in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p14/optimization_rounds_no_invariants/round2/output_model_new.py), centered on `raw_buffer_load_x4`, LDS staging in `u32` words, and BF16 operand reinterprets for `mfma_32x32x8_bf16_f32`. I was not able to obtain a correctness-passing result for this round. The required evaluation command fails during Substrate lowering with `builtin.unrealized_conversion_cast` on the MFMA operand path, including on the original starting kernel shape and on the revised `u32 -> view -> mfma` variants. The last command run was the exact required round-local eval command, and it exited with that lowering error rather than a correctness result.
- total_tokens: 2052065
- error: Substrate: Using AmdgpuDriver (available: ['AmdgpuDriver', 'NvidiaDriver']) Failed to load custom CUDA kernel; Compiled but not able to run, count as runtime error. Error: Failed to lower MLIR to LLVM IR: LLVM Translation failed for operation: builtin.unrealized_conversion_cast remaining translation blockers: loc("/tmp/tmprioahxcj.py":104:46): builtin.unrealized_conversion_cast loc("/tmp/tmprioahxcj.py":104:59): builtin.unrealized_conversion_cast loc("/tmp/tmprioahxcj.py":105:46): builtin.unr...
### round3
- status: eval_failed
- unlocked hints: 3
- agent: codex
- summary: Implemented a narrowed kernel in [output_model_new.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p14/optimization_rounds_no_invariants/round3/output_model_new.py), without modifying [candidate_input.py](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p14/optimization_rounds_no_invariants/round3/candidate_input.py). The required evaluation command runs past compilation now, but the kernel is still not correct. The latest round-local result in [debug_eval.jsonl](/workspace/kernel_benchmark/kb_eval_pipeline/runs/kb1/p14/optimization_rounds_no_invariants/round3/debug_eval.jsonl) reports `compiled: true` and `correctness: false` with `max_difference: 96` and `avg_difference: 4.90625`. I’m stopping here because this round did not reach the required correctness-passing state within the allowed time.
- total_tokens: 947847
- error: Output mismatch
