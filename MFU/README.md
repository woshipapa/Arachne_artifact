# MFU

Utilities for estimating FLOPs and MFU for HunyuanVideo workloads and schedule-based runs.

## Files
- calculate_dit_mfu.py: Estimate per-GPU FLOPs for HunyuanVideo DiT with sequence-parallel splits and fwd/bwd factors.
- calculate_vae_mfu.py: Estimate VAE encoder FLOPs using 3D conv, resnet, downsample, and attention formulas.
- new_total_mfu.py: Parse task names and schedules, combine with iteration times, and emit MFU summaries.
- pipeline_mfu.py: Simulate per-rank pipeline load (VAE temporal and spatial tiling plus DiT) to show imbalance.
- task_yaml_cal_mfu_flops.py: Parse schedule YAMLs, topologically order tasks, compute per-rank FLOPs JSON and CSV.
- __init__.py: Package exports for the FLOPs estimator helpers.

## Typical inputs and outputs
- Inputs: schedule YAMLs, per-iteration timing logs, and task metadata inferred from task names.
- Outputs: JSON files with per-rank FLOPs, and CSV summaries of per-iteration MFU.
