# tracing

Trace annotation. Currently an NVTX labeler abstraction with automatic
fallback.

## Where to look

1. Adding range annotations where NVTX is available
Use `create_labeler(preferred="nvtx")`

2. Keeping the same code path without NVTX
Use `create_labeler(preferred="auto")` (falls back to a no-op labeler)

## Minimal example

```python
from my_utils.tracing import create_labeler

labeler = create_labeler(preferred="auto")
with labeler.range("forward"):
    # ... forward ...
    pass
```

## Key files

- `nvtx_utils.py`: `LabelerProtocol`, `NoOpLabeler`, `NvtxLabeler`, `TorchNvtxLabeler`, `create_labeler`
