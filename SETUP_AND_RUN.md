# Arachne Artifact — Setup & Run Guide

This document covers everything needed to take the artifact from a fresh
checkout to a running training loop on the target training image, plus the
pitfalls found while validating it. It is written for artifact evaluation and
reproduction.

---

## 0. Prerequisites: what the image provides

The artifact does **not** bundle CUDA or the deep-learning base stack; it relies
on the training image for those. Pull the image on every machine that will take
part in the run:

```bash
docker pull ghcr.io/woshipapa/nvcr-torch2.7:sc26
```

The image supplies the base stack (CUDA, driver, a CUDA toolkit with `nvcc`);
`setup_pyenv.sh` then builds `.venv` on top of it. What matters at run time is
the venv, whose versions are those the reported experiments were run on:

| Component | Version | Where it comes from | Notes |
|---|---|---|---|
| Python | 3.10.12 | image (`/usr/bin/python`) | See the 3.12 syntax note in §5.4 |
| NVIDIA driver + CUDA | H100 / cu126 | image | At least one GPU must be attached |
| `nvcc` | CUDA 12.x | image | Needed to build `transformer_engine`; put it on `PATH` and set `CUDA_HOME` if the image does not |
| uv | 0.9+ | image | `setup_pyenv.sh` uses it to create the venv |
| torch | 2.7.1+cu126 | venv (`requirements.txt`) | |
| transformer_engine | 2.10.0 | venv (pinned in `setup_pyenv.sh`) | **Probes the driver at import time**; without a GPU it reports `0 active drivers` |
| diffusers | 0.34.0 | venv (`requirements.txt`) | |
| deepspeed | 0.16.2 | venv (`requirements.txt`) | |
| tensordict / xgboost / scikit-learn / joblib | 0.10.0 / 3.1.2 / 1.7.2 / 1.5.2 | venv (pinned in `setup_pyenv.sh`) | |

> These versions are pinned deliberately. `transformer_engine` in particular
> must not float: releases from 2.17 onward build against the PyTorch 2.8 header
> layout (`c10d/symm_mem/SymmetricMemory.hpp`) and fail to compile against the
> torch 2.7.1 pinned here, and an unpinned resolve also picks the CUDA 13 build
> instead of the CUDA 12 one matching `torch 2.7.1+cu126`.

> `transformer_engine` and `deepspeed` fail to import in a session with **no
> GPU** (`RuntimeError: 0 active drivers`). That is driver probing, **not a
> broken environment** — training must be validated on a GPU-attached machine.

---

## 1. Getting the code

```bash
git clone <artifact-repo-url> Arachne-artifact
cd Arachne-artifact
```

### 1.1 Vendored dependencies (no separate install needed)

Four packages required by training are **vendored into the repository** in
trimmed form. They were originally installed with `pip install -e` from paths
outside the repository, which do not survive relocation, so they are bundled
here to keep the artifact self-contained:

| Directory | Role | Resolved via |
|---|---|---|
| `vast/` | Datasets, models, pipelines, samplers | PYTHONPATH (repo root) |
| `teleai_data_tool/` | Dataset IO (lmdb / pkl / file datasets, schema) | PYTHONPATH (repo root) |
| `simulation_data/` | `IterationLogParser` (reads iteration logs for the dynamic-flex scheduler) plus the workload traces `simulation_log_*.txt` (hunyuan 720p/360p/1080p, wan, cogvideox) and `total_frames_*.txt`. Each trace is the sequence of per-iteration, per-rank tensor shapes recorded from bucket sampling on the **real** datasets, so all systems replay the identical real-world workload deterministically — no raw video needed | PYTHONPATH (repo root) |
| `my_utils/` | `GlobalLogger`, `DebuggingEvent`, `MyTimer`, profiling | PYTHONPATH (`<repo root>/my_utils`, nested package) |

The trained cost models the dynamic-flex scheduler needs (~21 MB) are bundled
as well:

| Directory | Role |
|---|---|
| `trained_models/{hunyuan,wan,wan14b,cogvideox}/{360p,720p,1080p}/` | The `*_analytical.joblib`, `*_xgb.json` and `*_ml_columns.joblib` files loaded by the VAE/DiT cost-model predictors (a two-stage "analytical baseline + XGBoost residual" model). Read when `SchedulePool` is initialised for genetic/dynamic scheduling. **Required.** |
| `t2v_flow/planner/generated_schedules/{MODEL_TYPE}/{RESOLUTION}/{MAX_FRAMES}/{SCHEDULE_TYPE}/schedule_{iteration}.yaml` | One pre-generated execution plan **per iteration** (per-rank task schedule), read by `handler.update_and_setup_for_iteration`. Covers hunyuan (360p/720p, several frame windows), wan and cogvideox, each across {flex_sp, genetic, megatron-lm}. |

> **Plans are bound to a specific GPU count.** The plans under
> `generated_schedules` are pre-generated for a fixed world size — for example
> hunyuan/wan/cogvideox 720p/129 assume **16 GPUs**, hunyuan/720p/4nodes assumes
> **32**, and hunyuan/360p/129 assumes **64**. A plan assigns tasks to ranks
> 0..N-1, so the run **must use the matching GPU count**. Running on fewer GPUs
> raises `KeyError: <rank>` because the plan references a rank that does not
> exist. That is a world-size mismatch, not a missing file. The standard
> configuration is 16 GPUs.

> `my_utils` is a **nested package** (`my_utils/my_utils/`), so PYTHONPATH needs
> the extra `<repo root>/my_utils` level. The launch scripts handle this (§2.3).

---

## 2. Environment preparation

### 2.1 Automatic (recommended): `setup_pyenv.sh`

The launch script `run_unified_many.sh` begins with
`source setup_pyenv.sh`, which calls `setup_env_and_install` itself; on a
non-zero return the launcher aborts rather than entering training.

The first run performs the steps below. A completion marker
`.venv/.arachne_setup_done` is written only after everything succeeds, and
later runs skip the install on the strength of it; an install that died partway
leaves an unmarked `.venv`, which is removed and rebuilt on the next run.

1. `uv venv -p /usr/bin/python --system-site-packages` creates `.venv`,
   inheriting the image's system packages (torch, TE, apex).
2. `uv pip install -r requirements.txt` (154 pinned dependencies, §2.2).
3. `uv pip install --no-build-isolation -e my_utils` (installs `my_utils` and
   `CtrlRandom`). The flag matters: `my_utils` builds with setuptools, which
   step 2 already put in the venv, and without it PEP 517 builds in a fresh
   isolated environment and refetches setuptools from the index — a needless
   network round-trip that fails the whole install on a slow link.
4. `uv pip install omegaconf einops yunchang-*.whl` then the pinned set:
   `transformer_engine[pytorch]==2.10.0`, `tensordict==0.10.0`, `etcd3`,
   `joblib==1.5.2`, `xgboost==3.1.2`, `scikit-learn==1.7.2`.
   `transformer_engine` is compiled here, so this is the slowest step and the
   one that needs `nvcc`.
5. **Required patch** — copy the custom VAEs into the installed `diffusers`:
   ```
   cp megatron/core/models/vae/autoencoder_kl*          .venv/.../site-packages/diffusers/models/autoencoders/
   cp megatron/core/models/vae/tile_parallel_utils.py   .venv/.../site-packages/diffusers/models/autoencoders/
   ```
   > This step is **mandatory** — without it the VAE behaves differently. The
   > script does it automatically.

### 2.2 `requirements.txt`

154 pinned public PyPI packages (torch, diffusers, transformers, lmdb, decord,
cattrs, loguru, yunchang, …). Reviewers only need
`pip install -r requirements.txt`; no internal package source is involved.

### 2.3 Import resolution: PYTHONPATH (set by the launch scripts)

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR:$SCRIPT_DIR/my_utils
```
- `$SCRIPT_DIR` resolves `vast`, `teleai_data_tool`, `simulation_data`, `megatron`
- `$SCRIPT_DIR/my_utils` resolves the nested inner `my_utils` package

---

## 3. Verifying the environment (two levels)

### 3.1 Dependency self-check: `check_env.py` (first four layers need **no GPU**)

```bash
python check_env.py
```

It verifies five layers in order: core dependencies → vendored packages
(`vast` / `teleai_data_tool` / `my_utils` / `simulation_data`) → vast training
submodules → Megatron core → the GPU layer
(`transformer_engine` / gpt / training).

- **Without a GPU**: the first four layers should all report `OK` and the GPU
  layer is skipped with a notice. This is expected.
- **With a GPU**: all five layers should report `OK`.

The script is self-contained — it adds the repo root and `my_utils` to
`sys.path` itself, so it can be run standalone.

### 3.2 Single-node smoke test: `run_single_node_test.sh` (**needs a GPU**)

```bash
bash run_single_node_test.sh
```
- Single node, no multi-node rendezvous (`WORLD_SIZE=1`, `MASTER_ADDR=127.0.0.1`)
- Fake batches (`USE_FAKE_BATCH=1`, no real data), `TRAIN_ITERS=3`
- Auto-detects the GPU count: `TP=1 CP=2` with 2+ GPUs, `TP=1 CP=1` with one
- Overridable: `CUDA_VISIBLE_DEVICES`, `TP`, `CP`, `TRAIN_ITERS`, `GBS`

**Success criterion**: model-construction logs
(`number of parameters ... 13067700243`) → VAE/DiT pipeline initialisation →
forward step and iteration logs. At that point the environment and code path
are fully ready.

> A single GPU is a degenerate case and prints many
> `no custom group assigned, fallback to static CONTEXT_PARALLEL_GROUP` lines.
> That is the **expected** fallback of the dynamic-flex scheduler to a static CP
> group on one GPU, not an error.

**Measured on 1×H100 80GB with the default 13B model**: Megatron init →
distributed init → dynamic-flex scheduler (multi-broker / prefetch / broker
threads) → VAE pipeline → full 13B DiT construction
(`number of parameters ... 13067700243`) → transfer to GPU, ending in
`CUDA out of memory` (79.14 / 79.18 GiB used) while placing the 13B weights and
optimizer state on a single card.

> That OOM is a **memory/GPU-count limit, not an environment or code problem** —
> the full 13B model is designed for multiple GPUs. Shrinking the model
> (`NUM_LAYERS=2 NUM_SINGLE_LAYERS=2`) gets past it and continues to cost-model
> loading, DitPredictor / DitMemoryPredictor initialisation, and per-iteration
> plan loading.

**The inherent limit of a single GPU**: the plans read during dynamic-flex
training (`generated_schedules/.../schedule_{iter}.yaml`) are generated for
**16 GPUs** (see the note in §1.1). On one GPU the scheduling stage raises
`KeyError: <rank>` because the plan references ranks 5, 15 and so on while only
rank 0 exists. **This is a world-size mismatch, not a missing file.**

> So the smoke test is worth running to confirm that environment, dependencies,
> vendored packages, cost models, model construction and scheduler
> initialisation are all in place — all of which pass — but **actually executing
> genetically scheduled training iterations requires the GPU count the plans
> were generated for** (16 for the standard hunyuan/720p configuration, i.e. two
> nodes × 8 GPUs, launched with `start_many.sh`).

---

## 4. Full multi-node training

Main entry point:
```bash
bash start_many.sh <NODE_RANK>
# internally: start_many.sh -> run_unified_many.sh <NODE_RANK> <TP> <CP> <FRAMES> <RES_W> <RES_H>
#                           -> torchrun ... pretrain_hunyuanvideo.py
```
Defaults: `GBS=8 USE_FAKE_BATCH=1 ... 1 2 49 1280 720` (TP=1, CP=2, 49 frames,
1280×720, hunyuan). A real multi-node run needs `MASTER_ADDR`, `WORLD_SIZE`
(= number of nodes) and each node's `NODE_RANK` set for the cluster.

Key tunables (all `${VAR:-default}` in `run_unified_many.sh`, overridable from
the environment): `MODEL_TYPE` (hunyuan/wan/cogvideox), `RESOLUTION`,
`NUM_LAYERS` / `NUM_SINGLE_LAYERS`, `SCHEDULE_TYPE`
(genetic/flex_sp/megatron-lm), `GBS`, `TRAIN_ITERS`, `CUDA_VISIBLE_DEVICES`,
and the various `ENABLE_*` communication-optimisation switches.

---

## 5. Troubleshooting

### 5.1 Reusing a venv copied from another machine

If you reuse a venv relocated from another machine instead of creating one with
`setup_pyenv.sh`, the shebang lines in its `bin/*` and the `VIRTUAL_ENV` value
in `bin/activate` may still point at the old absolute path, which causes:
- `source .venv/bin/activate` to silently do nothing (PATH is not rewritten, so
  the system python/torchrun is used)
- `torchrun` to fail with `bad interpreter: No such file or directory`

Fix by rewriting the old path:
```bash
OLD=<old venv absolute path>; NEW=$(pwd)/.venv
cd "$NEW/bin"
for f in $(grep -rl "$OLD" .); do
  case "$f" in
    ./activate*) sed -i "s|$OLD|$NEW|g" "$f" ;;
    *)           sed -i "1s|^#!.*python3\$|#!$NEW/bin/python3|" "$f" ;;
  esac
done
```
> Creating `.venv` fresh with `setup_pyenv.sh` avoids this entirely.

### 5.2 `transformer_engine` / `deepspeed` report `0 active drivers`

The session has **no GPU attached**; `transformer_engine` probes the driver at
import time. Move to a GPU-attached machine — `nvidia-smi -L` should list GPUs
and `ls /dev/nvidia*` should be non-empty.

### 5.3 `ModuleNotFoundError: No module named 'vast' / 'teleai_data_tool' / 'simulation_data' / 'my_utils'`

PYTHONPATH is missing the repo root (and the `my_utils` subdirectory). Launch
through the scripts, which set it, or export it manually:
`export PYTHONPATH=$PWD:$PWD/my_utils:$PYTHONPATH`.

### 5.4 `SyntaxError: f-string expression part cannot include a backslash` (Python 3.10)

The nsys **visualisation** modules under `my_utils`
(`profiling/sources/nsys_timeline_html.py` and others) use f-string syntax only
permitted from Python 3.12. Those modules are **not on the training path** and
are imported optionally (loaded on 3.12, skipped on 3.10), so base imports and
training are unaffected. To use the visualisation on 3.10, either rewrite the
f-strings or switch to a 3.12 interpreter.

### 5.5 `--dataloader-type` and real data

The smoke test uses `USE_FAKE_BATCH=1` and reads no real data. Data and model
paths in `hunyuanvideo_config.py` and similar files are written as explicit
placeholders that real training must be pointed at local locations:

- `<DATA_ROOT>` — the dataset root (e.g. the Text2Video annotation packs under
  `<DATA_ROOT>/Text2Video/annotations/...`).
- `<MODEL_ROOT>` — the pretrained model / HuggingFace hub root (e.g.
  `<MODEL_ROOT>/hunyuan/hunyuanvideo_13b`).
- `<CKPT_ROOT>` — checkpoints, text encoders and per-dataset manifest JSONs.

Replace these with your own paths (or point the corresponding environment
variables at them) before a real training run.

---

## 6. TL;DR

```bash
# 0) confirm the machine has a GPU (nvidia-smi -L produces output)
docker pull ghcr.io/woshipapa/nvcr-torch2.7:sc26
# 1) get the code
git clone https://github.com/woshipapa/Arachne_artifact.git
cd Arachne_artifact
# 2) build the venv, install deps, patch the VAE.
#    Use `source`, not `bash`: the script activates .venv in the calling
#    shell, and a subshell would leave step 3 on the system interpreter.
source setup_pyenv.sh
# 3) dependency self-check (first four layers pass without a GPU)
python check_env.py
# 4) single-node smoke test (needs a GPU) -- 13B params + VAE/DiT init + iterations
bash run_single_node_test.sh
# 5) full multi-node training
bash start_many.sh <NODE_RANK>
```
