#!/usr/bin/env bash
# One-time setup, on a VIZ NODE (barklaviz1.liv.ac.uk) — not the login node.
#
#   ssh <user>@barklaviz1.liv.ac.uk
#   cd /mnt/fastscratch/users/$USER/coder-agent
#   bash scripts/build_base_env.sh
#
# Builds the two things a compute node cannot build for itself:
#
#   1. the vLLM Apptainer image (no vLLM module exists on Barkla)
#   2. the pinned base environment every generated experiment inherits
#
# Both live on fastscratch. Neither can be created during a run: Barkla's
# guidance kills long builds on the login node, and compute nodes frequently
# have no outbound network at all — which is exactly why the base env carries
# the scientific stack up front instead of installing it per experiment.
set -euo pipefail

SCRATCH="${SCRATCH:-/mnt/fastscratch/users/$USER}"
BASE_ENV="${CODER_BASE_ENV:-$SCRATCH/coder-agent-base}"
CONTAINER_DIR="${CONTAINER_DIR:-$SCRATCH/containers}"
VLLM_SIF="${VLLM_SIF:-$CONTAINER_DIR/vllm.sif}"
VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:latest}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"

MODEL="${LLM_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"

echo "==> scratch:   $SCRATCH"
echo "==> base env:  $BASE_ENV"
echo "==> container: $VLLM_SIF"
echo "==> HF_HOME:   $HF_HOME"
echo

mkdir -p "$SCRATCH" "$CONTAINER_DIR" "$HF_HOME"

# --- 0. uv ------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "==> installing uv (Barkla provides no module for it)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# --- 1. the base environment for generated experiments ----------------------
# Every package a generated experiment is likely to reach for. Adding to this
# list is the correct fix when a run reports "cannot install X: no outbound
# network" — that message means the package was needed on a node that could not
# fetch it, and the answer is to have it here beforehand.
echo
echo "==> building the experiment base environment"
uv venv --python 3.11 "$BASE_ENV"
uv pip install --python "$BASE_ENV/bin/python" \
    numpy pandas scipy scikit-learn statsmodels matplotlib seaborn \
    pyarrow tqdm requests \
    geopandas shapely rasterio libpysal esda spreg \
    cmdstanpy pymc arviz xarray netCDF4 networkx

echo "==> base environment package count:"
uv pip freeze --python "$BASE_ENV/bin/python" | wc -l

# cmdstanpy ships no compiler toolchain; without this every Stan model in a
# generated experiment fails at build time on a node that cannot then fetch it.
echo "==> installing CmdStan into the base environment"
"$BASE_ENV/bin/python" -c "
import cmdstanpy, sys
try:
    cmdstanpy.install_cmdstan(overwrite=False, progress=False)
except Exception as exc:
    print(f'CmdStan install failed ({exc}); Stan-based experiments will fall back to PyMC', file=sys.stderr)
"

# --- 2. the vLLM container ---------------------------------------------------
if [[ -f "$VLLM_SIF" ]]; then
    echo
    echo "==> $VLLM_SIF already exists; skipping the container build"
else
    echo
    echo "==> building $VLLM_SIF (this takes a while)"
    export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SCRATCH/apptainer_cache}"
    export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$SCRATCH/apptainer_tmp}"
    mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
    apptainer build "$VLLM_SIF" "$VLLM_IMAGE"
fi

# --- 3. the model weights ----------------------------------------------------
# Apptainer only auto-binds $HOME; without --bind the container cannot write to
# an HF_HOME on fastscratch and the download dies with a read-only filesystem.
echo
echo "==> downloading $MODEL into $HF_HOME (~61GB for the 30B BF16 weights)"
apptainer exec --bind /mnt/fastscratch \
    --env HF_HOME="$HF_HOME" \
    "$VLLM_SIF" \
    hf download "$MODEL"

cat <<EOF

Done. Add these to your .env:

    CODER_BASE_ENV=$BASE_ENV
    LLM_MODEL=$MODEL

and export before submitting:

    export HF_HOME=$HF_HOME
    export VLLM_SIF=$VLLM_SIF
EOF
