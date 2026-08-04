# Arachne Artifact (Anonymous)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21783193.svg)](https://doi.org/10.5281/zenodo.21783193)

DeepSpeed baseline release: [10.5281/zenodo.21783264](https://doi.org/10.5281/zenodo.21783264) ([GitHub tag](https://github.com/woshipapa/Arachne_artifact/tree/deepspeed-baseline-v1.0.0)).

This repository is an anonymous artifact prepared for double-blind review.
It contains the code required to demonstrate the core training and execution pipeline.

## Scope
- Core source code for training/execution logic
- Launch scripts for end-to-end pipeline demonstration
- Minimal configuration and runtime dependencies

## Notes
- Non-essential assets and identifying materials have been removed.
- Paths may use placeholder-style dependency locations for artifact display purposes.
- `vast/` and `teleai_data_tool/` are vendored, minimal subsets of internal dependencies required to import and run the training pipeline. All Python dependencies (including theirs) are pinned in `requirements.txt`, installed automatically by `setup_pyenv.sh`.
- `setup_pyenv.sh` also copies custom VAE model files from `megatron/core/models/vae/` into the installed `diffusers` package (`diffusers/models/autoencoders/`) — this patch step is required and is applied automatically on first run.

## Entry
- Main launcher script: `start_many.sh` (multi-node training).
- **Setup & run guide: see [`SETUP_AND_RUN.md`](SETUP_AND_RUN.md)** for step-by-step
  environment preparation, verification, and troubleshooting on the target image.

## Single-machine testing
Two lightweight scripts are provided to verify the environment and that the
pipeline reaches the training loop on a single machine:

- **`python check_env.py`** — dependency / import self-check. Verifies that all
  Python dependencies and the vendored `vast` / `teleai_data_tool` packages are
  installed and importable, layer by layer. **Works without a GPU** (the
  GPU-only `transformer_engine` layer is reported separately and skipped when no
  driver is present). Run this first to confirm the environment is set up.

- **`bash run_single_node_test.sh`** — single-node training smoke test. Runs the
  real training entry (`run_unified_many.sh` → `pretrain_hunyuanvideo.py`) on
  **one machine only** (no multi-node rendezvous), with fake batches
  (`USE_FAKE_BATCH=1`) and a handful of iterations (`TRAIN_ITERS=3`), then exits
  cleanly. Seeing per-iteration training logs confirms the environment and the
  full code path are working. Requires at least 1 GPU. GPU count / parallelism /
  iterations are overridable via `CUDA_VISIBLE_DEVICES`, `TP`, `CP`,
  `TRAIN_ITERS`, `GBS`.

These test hooks are non-invasive: `run_unified_many.sh` was made to respect a
pre-set `CUDA_VISIBLE_DEVICES` and a `TRAIN_ITERS` override (both default to the
original values, so multi-node training behavior is unchanged).
