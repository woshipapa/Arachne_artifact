# artifacts

Offline artifacts: writing intermediate data to disk and reading it back, plus
NCU CSV analysis helpers.

## Where to look

1. Dumping tensors or intermediate results to disk
Use `UniversalDumper` / `DumpConfig`

2. Analysing and comparing NCU CSV metrics
Use `analyze_sm_throughput_from_csv` / `compare_kernel_metrics`

## Minimal example

```python
from my_utils.artifacts import DumpConfig, UniversalDumper

cfg = DumpConfig(output_dir="./dump_out")
dumper = UniversalDumper(cfg)
dumper.dump_tensor("x", x_tensor)
```

## Key files

- `dump_utils.py`: `DumpTensorIO`, `DumpConfig`, `UniversalDumper`, `UniversalLoader`
- `ncu_analyze_from_csv.py`: NCU CSV metric analysis and comparison
