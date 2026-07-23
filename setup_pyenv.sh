#!/bin/bash

# This file is sourced by run_unified_many.sh, so the wrapper below
# anchors to the repo root (every path here is relative to it) and restores the
# caller's working directory afterwards.
setup_env_and_install() {
    local _root _prev="$PWD" _rc
    _root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || {
        echo "failed to locate repo root"; return 1; }
    cd "$_root" || return 1
    _arachne_install_impl
    _rc=$?
    cd "$_prev" || return 1
    return $_rc
}

_arachne_install_impl() {
    # Keyed on a completion marker, not on the directory: a run that dies partway
    # through the install still leaves .venv behind, and skipping on that would
    # hand the training job a half-populated environment.
    #
    # This runs before the toolchain checks below on purpose. Every training
    # launch sources this file, and once the venv is built the install path is
    # never taken, so demanding uv here would block training on any shell where
    # uv is simply not on PATH.
    if [[ -f ".venv/.arachne_setup_done" ]]; then
        source .venv/bin/activate || { echo "failed to activate .venv"; return 1; }
        echo "venv exists; skipping install..."
        return 0
    fi

    # From here on an install really is going to happen, so the tools it needs
    # must be present.
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv not found on PATH; install it (https://docs.astral.sh/uv/) and rerun"
        return 1
    fi
    if [[ ! -x /usr/bin/python ]]; then
        echo "/usr/bin/python not found; the artifact expects the provided image"
        echo "(Python 3.10). Point setup_pyenv.sh at your 3.10 interpreter to override."
        return 1
    fi

    if [[ -d ".venv" ]]; then
        echo "found an incomplete .venv (no completion marker); reinstalling..."
        rm -rf .venv
    fi

    echo "Setting up environment and dependencies..."
    # The install runs in a subshell so `set -e` cannot leak into the sourcing
    # launcher. Its status is captured on the following line rather than tested
    # with `||` -- bash ignores `set -e` inside a compound command whose status
    # is being tested, which would make every step below non-fatal.
    (
        set -euo pipefail

        echo "1. Installing base dependencies..."
        echo "Creating virtual environment..."
        uv venv -p '/usr/bin/python' --system-site-packages
        source .venv/bin/activate
        uv pip install pybind11
        uv pip install -r requirements.txt

        echo "2. Virtual environment active"
        # vast/ and teleai_data_tool/ are vendored directly in this repo and are resolved
        # via PYTHONPATH by run_unified_many.sh, so they are not pip-installed here.
        echo "3. Installing project packages..."
        if [[ -d "my_utils" ]]; then
            echo "Installing package: my_utils"
            # --no-build-isolation: my_utils builds with setuptools, which
            # requirements.txt already pinned into this venv above. Without the
            # flag PEP 517 builds it in a fresh isolated environment and refetches
            # setuptools from the index, so a slow or flaky network fails the
            # install for a dependency that is already present.
            uv pip install --no-build-isolation -e my_utils
        else
            echo "package dir missing, skipping: my_utils"
        fi

        uv pip install omegaconf
        uv pip install yunchang-0.6.0-py3-none-any.whl
        uv pip install einops

        # Pinned to the versions the reported experiments ran on. Leaving these
        # unpinned makes the install depend on whatever is newest at the time,
        # which does not stay compatible: transformer_engine 2.17 builds against
        # the PyTorch 2.8 header layout (c10d/symm_mem/SymmetricMemory.hpp) and
        # fails to compile against the torch 2.7.1 pinned in requirements.txt,
        # and an unpinned resolve also selects the CUDA 13 build rather than the
        # CUDA 12 one that matches torch 2.7.1+cu126.
        uv pip install --no-build-isolation 'transformer_engine[pytorch]==2.10.0'

        uv pip install 'tensordict==0.10.0'
        uv pip install etcd3
        uv pip install 'joblib==1.5.2'
        uv pip install 'xgboost==3.1.2'
        uv pip install 'scikit-learn==1.7.2'

        SITE=$(python -c "import diffusers, os; print(os.path.dirname(diffusers.__file__))")
        cp -r megatron/core/models/vae/autoencoder_kl* "$SITE/models/autoencoders/"
        cp megatron/core/models/vae/tile_parallel_utils.py "$SITE/models/autoencoders/"
        touch .venv/.arachne_setup_done
        echo "----------------------------------------"
        echo "All dependencies installed."
    )
    local _install_rc=$?
    if (( _install_rc != 0 )); then
        echo "environment setup failed (exit $_install_rc); see log above"
        return 1
    fi

    source .venv/bin/activate || { echo "failed to activate .venv"; return 1; }
    echo "venv location: $(pwd)/.venv"
}


setup_env_and_install
