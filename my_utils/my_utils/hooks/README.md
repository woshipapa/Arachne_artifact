# hooks

Observation and scoped profiling control built on PyTorch hooks.

## Where to look

1. Starting/stopping a profiler on a training window
Use `ForwardProfilerHook`

2. Recording module-level forward inputs and outputs
Use `ForwardTraceRecorder`

3. Module-level timing (legacy approach)
Use `ModuleProfiler` (optional)

## Minimal example

```python
from my_utils.hooks import ForwardTraceRecorder

recorder = ForwardTraceRecorder()
recorder.register(model)
# the recorded results can be read back after the forward pass
```

## Key files

- `ForwardProfileHook.py`: `ForwardProfilerHook`
- `module_hook.py`: `ForwardTraceRecorder`
- `moduleProfiler.py`: `ModuleProfiler`

## Compatibility

- The legacy import paths `my_utils.ForwardProfileHook` and
  `my_utils.module_hook` still work.
