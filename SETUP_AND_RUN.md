# DeepSpeed Baseline — Setup & Run Guide

This branch holds the **DeepSpeed baseline** that Arachne is compared against: the
original `vast` T2V training framework running on DeepSpeed plus a customized
`accelerate`. It shares the same container image and the same virtualenv as the
Arachne artifact on the `main` branch.

---

## 0. Prerequisites: the same stack as the Arachne artifact

The baseline is the `deepspeed-baseline` branch of the **same repository** as the
Arachne artifact, so no second checkout is needed:

```bash
git checkout main
bash setup_pyenv.sh            # builds ./.venv (Python 3.10, torch 2.7.1+cu126,
                               # transformer_engine, deepspeed 0.16.2, ...)
git checkout deepspeed-baseline
bash start_many.sh 0           # reuses the same ./.venv
```

`.venv/` is untracked and gitignored, so it survives the branch switch. Building
it once on `main` is enough for both branches — `requirements.txt` there already
pins `deepspeed==0.16.2`, the version reported in the paper.

A GPU must be attached: `transformer_engine` probes for an active driver at
import time.

---

## 1. Layout

```
<repo root>  (branch: deepspeed-baseline)
├── start_many.sh                   # unified entry point (see §4)
├── setup_env.sh                    # VAE patch + my_utils, run automatically
├── teleai_data_tool/               # vendored data-IO package (same files as main)
├── simulation_log_*.txt            # per-model / per-resolution workload traces
└── vast3/
    ├── accelerate/                 # customized accelerate (source, on PYTHONPATH)
    │   └── src/accelerate/
    └── vast/                       # training code (note the two nested levels)
        ├── tools/train.py          # training entry
        ├── projects/<model>/configs/<config>.py
        ├── vast/                   # the vast package itself
        └── my_utils/               # my_utils package
```

Entry point: `python tools/train.py projects/<model>/configs/<config>.py`.
`launch_from_config` spawns the workers via `accelerate launch`; the process
count comes from `gpu_ids` in the config.

---

## 2. Environment preparation

### 2.1 `setup_env.sh` (run automatically by `start_many.sh`)

Run from the repo root — `bash setup_env.sh` — or just launch training and let
it happen. It:

1. activates `./.venv` (built on `main`; errors out with instructions if absent),
2. copies `vast3/vast/vast/models/vae/autoencoder_kl*` **and
   `tile_parallel_utils.py`** into the installed `diffusers`
   (`diffusers/models/autoencoders/`) — **required**,
3. installs `my_utils` in editable mode.

Steps 2–3 are recorded by `.venv/.ds_baseline_setup_done`, so repeated launches
skip them; a run that fails partway leaves no marker and is retried next time.

> **The tiling VAE is mandatory.** The four VAE sources under
> `vast3/vast/vast/models/vae/` are the tile-parallel versions, identical to the
> ones the Arachne artifact uses. The baseline imports `AutoencoderKLWan` from
> `diffusers`, and the wan config sets `vae_tiling=True` → `enable_tiling()` at
> runtime. Without the patch the VAE behaviour differs from Arachne's and the
> comparison is invalid.

### 2.2 PYTHONPATH (set by `start_many.sh`; four entries, all required)

```bash
R=<repo root>
export PYTHONPATH=$R/vast3/accelerate/src:$R/vast3/vast:$R/vast3/vast/my_utils:$R:$PYTHONPATH
```
- `accelerate/src` → the customized accelerate (**not** the pip-installed one;
  it must come first)
- `vast3/vast` → resolves `import vast` (the package is at `vast3/vast/vast/`)
- `vast3/vast/my_utils` → nested `my_utils` package
- repo root → `teleai_data_tool`, the vendored data-IO dependency

### 2.3 Source fixes (already applied on this branch)

Two changes are already in the tree; they are documented here only so the
behaviour is not surprising.

- **sageattention fallback** — `vast3/vast/vast/models/dit/wan_dit/WanModel.py`
  catches `ImportError` (not just `ModuleNotFoundError`) around
  `from sageattention import sageattn`. Under `accelerate launch`, a missing
  `sageattention` is resolved as a namespace package and raises `ImportError`,
  which would otherwise bypass the fallback and crash every rank.
  `vast.models.__init__` imports `wan_dit` unconditionally, so this affects
  every config, not just wan.
- **`ALLOW_MISSING_WEIGHTS`** — `vast3/vast/vast/models/utils/paths.py` returns
  the path instead of raising when a checkpoint is absent and
  `ALLOW_MISSING_WEIGHTS=1` is set, letting the model fall back to default
  (random) initialization. Unset, the original error behaviour is preserved.

---

## 3. Data and weights

| Item | Config key | What to do |
|---|---|---|
| Pretrained weights | `pretrained` | Point at a local checkpoint, or set `ALLOW_MISSING_WEIGHTS=1` for random init (§3.1) |
| Workload trace | read by the dataloader | Shipped in this repo as `simulation_log_<model>_<res>[...].txt`; selected automatically from `MODEL_TYPE` / `RESOLUTION` / `MAX_FRAMES` / `WORLD_SIZE`, or overridden with `SIM_LOG` |
| Training data | each config's data list | Real training needs a local dataset; smoke tests can run without one |

### 3.1 Running without weights

```bash
export ALLOW_MISSING_WEIGHTS=1
```
VAE, transformer and checkpoint are then constructed with default (random)
initialization and the run proceeds to DeepSpeed engine setup. This is enough to
reproduce the per-iteration timing the baseline is measured on; it is not a
convergence run.

---

## 4. Running

`start_many.sh` is the single entry point, matching the `main` branch's
convention — one node rank per invocation, everything else via environment
variables:

```bash
MODEL_TYPE=hunyuan   RESOLUTION=720p bash start_many.sh <node_rank>
MODEL_TYPE=cogvideox RESOLUTION=720p bash start_many.sh <node_rank>
MODEL_TYPE=wan1.3b   RESOLUTION=720p bash start_many.sh <node_rank>
```

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_TYPE` | `hunyuan` | `hunyuan` \| `cogvideox` \| `wan` / `wan1.3b` |
| `RESOLUTION` | `720p` | `360p` \| `720p` \| `1080p` |
| `MAX_FRAMES` | auto | Frame window (1080p → 57, otherwise 129) |
| `WORLD_SIZE` | `2` | Number of **nodes** (8 GPUs each). 4 or 8 selects the node-scaling workload |
| `MASTER_ADDR` / `MASTER_PORT` | `127.0.0.1` / `12346` | Rendezvous |
| `SIM_LOG` | auto | Explicit workload-trace override |
| `ALLOW_MISSING_WEIGHTS` | unset | `1` → random init (§3.1) |

The GPU count comes from `gpu_ids` in the config (default 8). Lower it for a
single-GPU smoke test.

---

## 5. Troubleshooting

- **`ModuleNotFoundError: No module named 'teleai_data_tool.logger'`**
  → repo root missing from PYTHONPATH (§2.2). Launch via `start_many.sh`.
- **`ImportError: cannot import name 'sageattn' from 'sageattention'`**
  → the fallback in §2.3 was reverted, or install `sageattention`.
- **`No module named 'vast'`, or accelerate resolving to the official package**
  → PYTHONPATH order: the customized `accelerate/src` must come first (§2.2).
- **`ValueError: <path> does not exist`**
  → `pretrained` / data paths point somewhere that does not exist; see §3.
- **8 processes spawned but fewer GPUs → ranks crash**
  → reduce `gpu_ids` in the config to the GPUs actually available.
- **`transformer_engine: 0 active drivers`**
  → no GPU attached to the session.
- **`No completed .venv found`**
  → build it once on `main` with `setup_pyenv.sh` (§0).
