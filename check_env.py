#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment self-check for the Arachne artifact.

This script verifies that all Python dependencies and the vendored
`vast` / `teleai_data_tool` packages are installed and importable, WITHOUT
needing a GPU. It imports the training stack layer by layer and prints a
clear PASS/FAIL for each layer so you can tell exactly where (if anywhere)
the environment is broken.

Note: `transformer_engine` (and therefore the full model-construction path)
requires a live NVIDIA driver and will fail on a CPU-only machine with
"0 active drivers". That is expected off-GPU and is reported separately as
INFO, not as a failure of the environment setup itself.

Usage:
    python check_env.py
"""
import importlib
import os
import sys

# Make the check self-contained: ensure the repo root (for vast / teleai_data_tool
# / megatron) and the vendored my_utils package dir are importable, mirroring what
# run_unified_many.sh puts on PYTHONPATH.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "my_utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# (display name, import statement) — ordered from leaf deps up to the
# training entry stack. Each layer builds on the previous one.
LAYERS = [
    ("1. core third-party deps", [
        ("torch", "import torch"),
        ("numpy", "import numpy"),
        ("yaml (pyyaml)", "import yaml"),
        ("einops", "import einops"),
        ("diffusers", "import diffusers"),
        ("transformers", "import transformers"),
        ("decord", "import decord"),
        ("lmdb", "import lmdb"),
        ("cattrs", "import cattrs"),
        ("loguru", "import loguru"),
        ("yunchang", "import yunchang"),
    ]),
    ("2. vendored packages", [
        ("vast", "import vast"),
        ("teleai_data_tool", "import teleai_data_tool"),
        ("my_utils (GlobalLogger/DebuggingEvent)", "from my_utils import GlobalLogger, DebuggingEvent, global_timer"),
        ("simulation_data (IterationLogParser)", "from simulation_data import IterationLogParser"),
    ]),
    ("3. vast submodules used by training", [
        ("vast.train.configs.config", "from vast.train.configs.config import load_config"),
        ("vast.datasets.config.t2v_200w", "from vast.datasets.config.t2v_200w import get_data_list"),
        ("vast.datasets.datasets.build", "from vast.datasets.datasets.build import build_dataset"),
        ("vast.datasets.transforms", "from vast.datasets.transforms import build_transform"),
        ("vast.models", "from vast.models import GuiderModel, HunyuanVideoTransformer3DModel, ModuleDict"),
        ("vast.pipelines", "from vast.pipelines import HunyuanVideoPipeline"),
        ("vast.train.samplers", "from vast.train.samplers import build_sampler"),
        ("vast.utils.acceleration", "from vast.utils.acceleration import parallel_states"),
        ("vast.datasets.bucket_config", "from vast.datasets.bucket_config import bucket, bucket_utils, read_cfg"),
    ]),
    ("4. Megatron core (up to parallel_state / yunchang)", [
        ("megatron.core", "import megatron.core"),
        ("megatron.core.tensor_parallel", "import megatron.core.tensor_parallel"),
        ("megatron.core.parallel_state", "import megatron.core.parallel_state"),
    ]),
]

# This layer needs a GPU driver (transformer_engine imports a Triton backend
# that probes for active CUDA drivers at import time).
GPU_LAYER = [
    ("transformer_engine", "import transformer_engine"),
    ("megatron.core.models.gpt", "import megatron.core.models.gpt"),
    ("megatron.training", "import megatron.training"),
]

GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
NC = "\033[0m"


def run_layer(title, checks):
    print(f"\n{BLUE}==== {title} ===={NC}")
    ok = True
    for name, code in checks:
        try:
            exec(code, {})
            print(f"  {GREEN}OK  {NC} {name}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  {RED}FAIL{NC} {name}  ->  {type(e).__name__}: {str(e)[:160]}")
    return ok


def main():
    print(f"{BLUE}Arachne artifact environment self-check{NC}")
    print(f"python: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")

    all_ok = True
    for title, checks in LAYERS:
        all_ok &= run_layer(title, checks)

    # GPU-dependent layer: report but do not count as environment failure.
    print(f"\n{BLUE}==== 5. GPU-dependent (needs NVIDIA driver) ===={NC}")
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        has_gpu = False

    if not has_gpu:
        print(f"  {YELLOW}INFO{NC} no usable GPU/driver; skipping transformer_engine etc.")
        print(f"       (not an environment failure -- rerun on a GPU node to verify this layer)")
    else:
        gpu_ok = run_layer("5. GPU-dependent imports", GPU_LAYER)
        all_ok &= gpu_ok

    print()
    if all_ok:
        print(f"{GREEN}==== Self-check passed: all non-GPU deps and vendored packages import ===={NC}")
        if not has_gpu:
            print(f"{YELLOW}     (GPU layer untested; recheck on a GPU node){NC}")
        sys.exit(0)
    else:
        print(f"{RED}==== Self-check found problems: inspect the FAIL items above ===={NC}")
        sys.exit(1)


if __name__ == "__main__":
    main()
