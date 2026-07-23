# distributed

Helpers for distributed training.

## Where to look

1. Aligning clocks across ranks
Use `ClockSynchronizer`

2. An etcd barrier
Use `etcd_barrier` (optional dependency: `etcd3`)

3. Sequence-parallel padding
Use `pad_for_sequence_parallel` / `remove_pad_by_value`
(optional dependency: `megatron.core`)

## Minimal example

```python
from my_utils.distributed import ClockSynchronizer

sync = ClockSynchronizer()
offset = sync.sync_once()
print("clock offset us:", offset)
```

## Key files

- `clockSyncUtils.py`: `ClockSynchronizer`, `SocketClockSynchronizer`
- `etcd_utils.py`: `etcd_barrier`
- `pad.py`: sequence-parallel padding helpers

## Note

- The `etcd` and `megatron` features are optional imports; the other modules
  work without those dependencies installed.
