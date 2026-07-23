#!/bin/bash
#
# DeepSpeed-baseline environment setup.
#
# The baseline shares the Arachne artifact's virtualenv: both branches live in
# the same repository, so the `.venv` built by `setup_pyenv.sh` on the `main`
# branch is still in the working tree after `git checkout deepspeed-baseline`
# (it is untracked and gitignored). This script reuses it and applies the two
# baseline-specific steps on top: the tile-parallel VAE patch and `my_utils`.
#
# Usage (from anywhere):  bash setup_env.sh
# `start_many.sh` runs it automatically, so a manual call is only needed if you
# want to prepare the environment ahead of the first training launch.

set -euo pipefail

cd "$(dirname "$0")"

GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'

if [[ ! -f ".venv/.arachne_setup_done" ]]; then
    echo -e "${RED}No completed .venv found in $(pwd).${NC}"
    echo "The baseline reuses the Arachne artifact's virtualenv. Build it once:"
    echo "    git checkout main && bash setup_pyenv.sh"
    echo "    git checkout deepspeed-baseline"
    echo "(requirements.txt on main pins deepspeed 0.16.2 and the rest of the stack.)"
    exit 1
fi

source .venv/bin/activate

# Marker for the baseline-specific steps below, kept separate from the main
# branch's .arachne_setup_done so each half can be redone independently.
if [[ -f ".venv/.ds_baseline_setup_done" ]]; then
    echo -e "${GREEN}DeepSpeed-baseline setup already applied; skipping.${NC}"
    exit 0
fi

echo "Configuring the DeepSpeed-baseline environment..."

# 1/2 -- VAE patch. The baseline imports AutoencoderKL* from diffusers, so the
# tile-parallel versions must replace the stock ones in the installed package.
# Resolve the target from the imported module rather than site.getsitepackages()
# so it always tracks the diffusers this interpreter actually loads.
echo "    - 1/2: patching diffusers with the tile-parallel VAEs..."
SRC_VAE_DIR="vast3/vast/vast/models/vae"
if ! ls "$SRC_VAE_DIR"/autoencoder_kl* >/dev/null 2>&1; then
    echo -e "${RED}    source VAEs not found under ${SRC_VAE_DIR}/${NC}"
    exit 1
fi
SITE=$(python -c "import diffusers, os; print(os.path.dirname(diffusers.__file__))")
TARGET_DIR="${SITE}/models/autoencoders"
if [[ ! -d "$TARGET_DIR" ]]; then
    echo -e "${RED}    diffusers target dir not found: ${TARGET_DIR}${NC}"
    exit 1
fi
cp -r "$SRC_VAE_DIR"/autoencoder_kl* "$TARGET_DIR/"
cp "$SRC_VAE_DIR/tile_parallel_utils.py" "$TARGET_DIR/"
echo "        patched: $TARGET_DIR"

# 2/2 -- my_utils (nested package, same one the main branch installs).
echo "    - 2/2: installing my_utils (editable)..."
if [[ ! -d "vast3/vast/my_utils" ]]; then
    echo -e "${RED}    vast3/vast/my_utils not found${NC}"
    exit 1
fi
if command -v uv >/dev/null 2>&1; then
    uv pip install -e vast3/vast/my_utils
else
    pip install -e vast3/vast/my_utils
fi

touch .venv/.ds_baseline_setup_done
echo -e "${GREEN}DeepSpeed-baseline environment ready.${NC}"
