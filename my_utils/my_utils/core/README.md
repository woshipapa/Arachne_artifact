# core

Base utilities. Independent of any profiling backend.

## Where to look

1. Timing and logging
See `utils.py` + `logger.py`

2. Temporarily patching a method
See `method_patch.py`

3. General debugging (tensors, checksums)
See `utils.py`

## Minimal example

```python
from my_utils.core import setup_logging_and_timer

logger, timer = setup_logging_and_timer(
    logger_name="train",
    log_file="train.log",
    use_cuda=True,
    rank=0,
)

timer.start("step")
# ... training code ...
timer.stop("step")
```

## Key files

- `utils.py`: `MyTimer`, `NoOpMyTimer`, `ChecksumUtils`, debugging helpers
- `logger.py`: `GlobalLogger`, `get_global_logger`
- `method_patch.py`: `MethodPatcher`, `MethodPatchHandle`
- `annotations.py`: `parametrize_shapes`

## Common imports

```python
from my_utils.core import MyTimer, NoOpMyTimer, setup_logging_and_timer
from my_utils.core import GlobalLogger, get_global_logger
from my_utils.core import MethodPatcher
```
