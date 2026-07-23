# legacy_profilers

Compatibility layer for the older profilers, kept so existing call sites keep
working.

## Where to look

1. Code still using the DITProfiler semantics
Use `create_profiler_context`

2. Code using the torch.profiler wrapper directly
Use `ProfilerWrapper`

## Key files

- `DITProfiler.py`: `create_profiler_context`
- `profilerwrapper.py`: `ProfilerWrapper`
