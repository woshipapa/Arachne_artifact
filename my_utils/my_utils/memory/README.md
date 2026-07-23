# memory

Memory diagnostics: snapshots, OOM signalling, GPU memory/utilisation tracking.

## Where to look

1. Capturing a memory snapshot at a key stage
Use `global_snapshotter`

2. Propagating an OOM flag across ranks
Use `set_oom_flag` / `check_oom_flag`

3. Monitoring GPU memory and utilisation
Use `GPU_Performance_Tracker` (optional dependency: `pynvml`)

## Minimal example

```python
from my_utils.memory import global_snapshotter, set_oom_flag, check_oom_flag

global_snapshotter.snapshot("before_step")
# ... train step ...
if check_oom_flag():
    print("OOM detected")
```

## Key files

- `memory_snapshot.py`: `MemorySnapshotter`, `global_snapshotter`
- `oom_restore.py`: OOM flag set/check
- `gpu_mem_tracker.py`: `GPU_Performance_Tracker`

## Note

- Enable tracking and snapshots only when needed; they add overhead to long
  training runs.
