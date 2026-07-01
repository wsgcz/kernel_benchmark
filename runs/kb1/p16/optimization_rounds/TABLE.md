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
| 1 | 1 | completed | True | True | 0.0143 | 1252950 | 0.249 | 17.400 |
| 2 | 2 | completed | True | True | 0.0326 | 1325898 | 0.270 | 8.270 |
| 3 | 3 | completed | True | True | 0.0311 | 221684 | 0.253 | 8.140 |

## Round Notes
### round0
- status: seeded
- agent: seed-copy
- summary: Seed round copied from the highest existing source round.
### round1
- status: completed
- unlocked hints: 1
- agent: codex
- summary: Implemented the kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p16/optimization_rounds/round1/output_model_new.py). Validation with the required round-local command passed correctness: `correctness: true` in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p16/optimization_rounds/round1/debug_eval.jsonl). The measured candidate runtime was about `17.1 us`.
- total_tokens: 1252950
### round2
- status: completed
- unlocked hints: 2
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p16/optimization_rounds/round2/output_model_new.py). It now uses a 4-wave `64x64` MFMA tile, stages A and B through LDS with `raw_buffer_load_x4`, packs each lane’s `16B` fragment into two natural BF16 MFMA steps, and uses double-buffered K16 staging with a 2-tile unrolled pipeline. Validation passed with the exact required command. The round-local debug result in [debug_eval.jsonl](/workspace/kb_eval_pipeline/runs/kb1/p16/optimization_rounds/round2/debug_eval.jsonl) reports `correctness: true`.
- total_tokens: 1325898
### round3
- status: completed
- unlocked hints: 3
- agent: codex
- summary: Implemented the optimized kernel in [output_model_new.py](/workspace/kb_eval_pipeline/runs/kb1/p16/optimization_rounds/round3/output_model_new.py). I preserved the existing pipelined/double-buffered MFMA kernel and kept the branchless ranged `raw_buffer_load_x4` path in place; the final bf16 writeback remains direct tensor indexing because `raw_buffer_store_x1` in this environment only accepts 32-bit payloads. Validation passed with the required command. The round-local eval reported `compiled: true`, `correctness: true`, and `runtime_us: 7.87`.
- total_tokens: 221684
