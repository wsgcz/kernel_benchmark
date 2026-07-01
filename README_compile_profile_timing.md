# Compile/Profile Timing Instrumentation

This note describes the changes needed to record wall-clock generation time,
compile/JIT validation time, and profiling time for the optimization-loop runs.
The goal is to answer reviewer questions without changing benchmark behavior.

## Output Schema

Each non-seed round should write round-level timing under:

```json
{
  "optimization_loop": {
    "timing": {
      "round_wall_time_s": 529.123,
      "agent_wall_time_s": 511.456,
      "final_eval_wall_time_s": 16.987,
      "orchestration_overhead_s": 0.680
    }
  }
}
```

The evaluator payload should include harness-internal timings under:

```json
{
  "eval_result": {
    "timings_s": {
      "load_reference_model_s": 0.120,
      "custom_model_import_compile_s": 1.230,
      "instantiate_and_to_device_s": 0.450,
      "correctness_check_s": 4.560,
      "first_candidate_invocation_s": 3.900,
      "candidate_profile_wall_time_s": 2.100,
      "reference_profile_wall_time_s": 1.080,
      "cleanup_s": 0.070,
      "eval_total_wall_time_s": 9.610,
      "compile_time_s": 5.130,
      "profile_time_s": 3.180
    }
  }
}
```

Recommended definitions:

- `round_wall_time_s`: end-to-end wall-clock time for one optimization round,
  from prompt creation until `meta.json` is ready to write.
- `agent_wall_time_s`: the agent subprocess duration. This includes model
  generation and any commands the agent runs internally.
- `final_eval_wall_time_s`: the orchestrator's post-agent validation/profiling
  subprocess duration.
- `orchestration_overhead_s`: `round_wall_time_s - agent_wall_time_s -
  final_eval_wall_time_s`, clamped at zero for reporting.
- `custom_model_import_compile_s`: host-side time spent importing/loading the
  generated `ModelNew` module. This catches eager compile/import work.
- `first_candidate_invocation_s`: the first `ModelNew(*inputs)` call during
  correctness. For Substrate/Triton-style lazy JITs, this is an upper bound for
  lazy compilation plus first execution.
- `compile_time_s`: `custom_model_import_compile_s +
  first_candidate_invocation_s`. In the paper, describe this as a conservative
  lazy-JIT compile upper bound, not pure compiler time.
- `candidate_profile_wall_time_s`: wall-clock time spent timing the generated
  candidate kernel after correctness passes.
- `reference_profile_wall_time_s`: wall-clock time spent timing the reference
  model for comparison.
- `profile_time_s`: `candidate_profile_wall_time_s +
  reference_profile_wall_time_s`.

## File 1: `optimization_loop/run_optimization_loop.py`

Add `import time` near the other standard-library imports.

In `run_single_round(...)`, add monotonic timers around the agent and final
evaluation stages:

```python
started_at = utc_now()
round_perf_start = time.perf_counter()

agent_perf_start = time.perf_counter()
try:
    agent_result = invoke_agent(...)
except subprocess.TimeoutExpired as exc:
    ...
except Exception:
    ...
agent_wall_time_s = time.perf_counter() - agent_perf_start

eval_wall_time_s = 0.0
if agent_result["returncode"] == 0:
    eval_perf_start = time.perf_counter()
    eval_payload, eval_exit_code = evaluate_round(args, current_round_dir)
    eval_wall_time_s = time.perf_counter() - eval_perf_start
    eval_was_run = True
else:
    ...

finished_at = utc_now()
round_wall_time_s = time.perf_counter() - round_perf_start
timing = {
    "round_wall_time_s": round(round_wall_time_s, 6),
    "agent_wall_time_s": round(agent_wall_time_s, 6),
    "final_eval_wall_time_s": round(eval_wall_time_s, 6),
    "orchestration_overhead_s": round(
        max(0.0, round_wall_time_s - agent_wall_time_s - eval_wall_time_s), 6
    ),
}
```

Then pass `timing=timing` into `build_round_meta(...)`.

Update `build_round_meta(...)` to accept a `timing: dict[str, Any]` argument and
write it into:

```python
"optimization_loop": {
    ...
    "started_at_utc": started_at,
    "finished_at_utc": finished_at,
    "timing": timing,
    ...
}
```

This provides the per-round wall-clock accounting for agent generation and final
orchestrator validation/profiling.

## File 2: `harness/evaluator/kernelbench/eval.py`

Add `import time`.

Extend `KernelExecResult` with a timing field:

```python
from pydantic import BaseModel, Field

class KernelExecResult(BaseModel):
    ...
    timings_s: dict = Field(default_factory=dict)
```

Inside `eval_kernel_against_ref(...)`, create a timing dictionary and helper:

```python
eval_total_start = time.perf_counter()
timings_s: dict[str, float] = {}

def finish_timing(name: str, start: float) -> None:
    timings_s[name] = round(time.perf_counter() - start, 6)

def finalize_result(result: KernelExecResult, context, device, tempfile):
    cleanup_start = time.perf_counter()
    graceful_eval_cleanup(context, device, tempfile)
    finish_timing("cleanup_s", cleanup_start)
    timings_s["eval_total_wall_time_s"] = round(
        time.perf_counter() - eval_total_start, 6
    )
    timings_s["compile_time_s"] = round(
        timings_s.get("custom_model_import_compile_s", 0.0)
        + timings_s.get("first_candidate_invocation_s", 0.0),
        6,
    )
    timings_s["profile_time_s"] = round(
        timings_s.get("candidate_profile_wall_time_s", 0.0)
        + timings_s.get("reference_profile_wall_time_s", 0.0),
        6,
    )
    result.timings_s = timings_s
    return result
```

Initialize `tempfile = None` before the custom model import/compile block so
early returns can always call `finalize_result(...)`.

Wrap the existing stages:

```python
stage_start = time.perf_counter()
Model, get_init_inputs, get_inputs = load_original_model_and_inputs(...)
...
finish_timing("load_reference_model_s", stage_start)

stage_start = time.perf_counter()
try:
    ...
    ModelNew, tempfile = load_custom_model_with_tempfile(...)
    ...
    torch.cuda.synchronize(device=device)
    finish_timing("custom_model_import_compile_s", stage_start)
except Exception:
    ...
```

Similarly wrap:

- `instantiate_and_to_device_s`: `ModelNew(*init_inputs)`, `.to(device=...)`,
  and synchronization.
- `correctness_check_s`: the `run_and_check_correctness(...)` call.
- `candidate_profile_wall_time_s`: candidate timing through
  `timing_fn(model_new, inputs, ...)` and `get_timing_stats(...)`.
- `reference_profile_wall_time_s`: reference timing through
  `timing_fn(original_model, inputs, ...)` and `get_timing_stats(...)`.

To record `first_candidate_invocation_s`, update `run_and_check_correctness(...)`
to accept an optional `timings_s` dict. Around the first `model_new(*inputs)`
call only:

```python
first_candidate_invocation_recorded = "first_candidate_invocation_s" in timings_s
...
if timings_s is not None and not first_candidate_invocation_recorded:
    first_start = time.perf_counter()
    output_new = model_new(*inputs)
    torch.cuda.synchronize(device=device)
    timings_s["first_candidate_invocation_s"] = round(
        time.perf_counter() - first_start, 6
    )
    first_candidate_invocation_recorded = True
else:
    output_new = model_new(*inputs)
    torch.cuda.synchronize(device=device)
```

All early returns in `eval_kernel_against_ref(...)` should return through
`finalize_result(...)`, except the existing retryable compilation race path that
returns `None`.

## File 3: `harness/tools/run_kernelbench_case.py`

When converting `KernelExecResult` to the JSONL payload, add:

```python
"timings_s": result.timings_s,
```

near `runtime_stats` and `ref_runtime_stats`:

```python
return {
    ...
    "runtime_stats": result.runtime_stats,
    "ref_runtime_stats": result.ref_runtime_stats,
    "timings_s": result.timings_s,
}
```

If `result is None`, optionally include an empty timing dictionary:

```python
"timings_s": {},
```

## Conv/Pytest Path

`evaluate_conv_round(...)` currently runs pytest directly from the optimization
loop rather than through `eval_kernel_against_ref(...)`. Add a simple subprocess
timer there:

```python
eval_start = time.perf_counter()
try:
    completed = subprocess.run(...)
finally:
    eval_wall_time_s = time.perf_counter() - eval_start
```

Write:

```json
"timings_s": {
  "eval_total_wall_time_s": eval_wall_time_s,
  "pytest_wall_time_s": eval_wall_time_s,
  "compile_time_s": null,
  "profile_time_s": null
}
```

for the conv pytest payload. This keeps the schema present even though pytest
does not expose compile/profile sub-stages.

## Aggregation for the Paper

For each kernel:

- `per_kernel_wall_clock_s = sum(round.optimization_loop.timing.round_wall_time_s)`
- `generation_time_s = sum(round.optimization_loop.timing.agent_wall_time_s)`
- `final_validation_profile_s =
  sum(round.optimization_loop.timing.final_eval_wall_time_s)`
- `compile_time_s = sum(round.eval_result.timings_s.compile_time_s)`
- `profile_time_s = sum(round.eval_result.timings_s.profile_time_s)`
- `iterations = number of non-seed completed round directories`

Report median, mean, p25-p75, and max across kernels. For the reviewer response,
state that `compile_time_s` is a conservative upper bound for lazy-JIT systems
because first execution can include compilation.

## Smoke Test

After applying the code changes, run a small existing case with one round or run
the harness directly on a single round directory. Then confirm:

```bash
python -m json.tool path/to/roundN/meta.json >/dev/null
```

and check that:

- `optimization_loop.timing.round_wall_time_s` exists.
- `optimization_loop.timing.agent_wall_time_s` exists.
- `eval_result.timings_s.eval_total_wall_time_s` exists.
- `eval_result.timings_s.compile_time_s` exists for KernelBench harness cases.
- `eval_result.timings_s.profile_time_s` exists when performance measurement is
  enabled.
