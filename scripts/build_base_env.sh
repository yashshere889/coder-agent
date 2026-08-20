#!/usr/bin/env bash
# One-time setup. Run on a VIZ NODE, not the login node:
#
#   ssh <user>@barklaviz1.liv.ac.uk
#   cd /mnt/fastscratch/users/$USER/coder-agent
#   bash scripts/build_base_env.sh
#
# Barkla §4: barklalogin1 is for editing and submitting only — "run lengthy
# builds or data transfers on barklaviz1/barklaviz2", and work overloading the
# login node "will be terminated without warning". The viz nodes allow up to 8
# cores and roughly 8 hours, which a container build and a 61GB download fit
# inside comfortably.
#
# Builds the three things a compute node cannot build for itself:
#
#   1. the vLLM Apptainer image   (no vLLM module exists on Barkla)
#   2. the pinned base environment every generated experiment inherits
#   3. the model weights in HF_HOME
#
# All three must exist beforehand because compute nodes frequently have no
# outbound network — which is also why a package missing at run time reports
# "add it here and rebuild" rather than retrying an unreachable index.
set -euo pipefail

FASTSCRATCH="/mnt/fastscratch/users/$USER"
BASE_ENV="${CODER_BASE_ENV:-$FASTSCRATCH/coder-agent-base}"
CONTAINER_DIR="${CONTAINER_DIR:-$FASTSCRATCH/containers}"
VLLM_SIF="${VLLM_SIF:-$CONTAINER_DIR/vllm.sif}"
VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:latest}"
MODEL="${LLM_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
export HF_HOME="${HF_HOME:-$FASTSCRATCH/hf_cache}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$FASTSCRATCH/apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$FASTSCRATCH/apptainer_tmp}"

echo "==> host:      $(hostname)"
echo "==> base env:  $BASE_ENV"
echo "==> container: $VLLM_SIF"
echo "==> HF_HOME:   $HF_HOME"
echo

case "$(hostname -s)" in
    barklaviz*) ;;
    barklalogin*)
        echo "REFUSING to run on the login node — Barkla §4 forbids lengthy builds here." >&2
        echo "ssh barklaviz1.liv.ac.uk and run this there." >&2
        exit 1 ;;
    *) echo "note: not on a recognised viz node; continuing anyway" ;;
esac

# --- 0. quota check ----------------------------------------------------------
# fastscratch is 500GB / 500k files (§5.1). This setup consumes roughly 61GB of
# weights, ~10GB of container, and the base env's ~100k inodes — so it fits, but
# only just, and finding that out after a 61GB download is a bad way to learn it.
echo "==> current fastscratch usage"
lfs quota -h -u "$USER" /mnt/fastscratch || true
echo

# --- 1. uv -------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "==> installing uv (Barkla provides no module for it)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# --- 2. the base environment for generated experiments -----------------------
# Every package a generated experiment is likely to reach for. Adding to this
# list is the correct fix when a run reports "cannot install X: no outbound
# network" — that message means the package was needed on a node that could not
# fetch it, and the answer is to have it here beforehand.
#
# It lives on fastscratch, not home: §5.1 says explicitly to avoid installing
# Python environments in home (75GB / 100k files) and that fastscratch is
# "suitable for installing Anaconda/Miniconda/Miniforge, Python and R".
echo
echo "==> building the experiment base environment"
# --allow-existing so this script can be re-run. It is the natural thing to do
# after adding a package to the list below, and `set -e` plus uv's "a virtual
# environment already exists" error otherwise aborts before the install step —
# leaving whatever partial package set the previous run managed.
uv venv --python 3.11 --allow-existing "$BASE_ENV"
uv pip install --python "$BASE_ENV/bin/python" \
    numpy pandas scipy scikit-learn statsmodels matplotlib seaborn \
    pyarrow tqdm requests \
    geopandas shapely rasterio libpysal esda spreg \
    cmdstanpy pymc arviz xarray netCDF4 networkx

echo "==> base environment holds $(uv pip freeze --python "$BASE_ENV/bin/python" | wc -l) packages"

# cmdstanpy ships no compiler toolchain. Without this, every Stan model in a
# generated experiment fails at build time on a node that cannot then fetch it.
echo "==> installing CmdStan into the base environment"
"$BASE_ENV/bin/python" - <<'PY' || true
import sys
try:
    import cmdstanpy
    cmdstanpy.install_cmdstan(overwrite=False, progress=False)
    print("CmdStan installed")
except Exception as exc:
    print(f"CmdStan install failed ({exc}); Stan experiments will fall back to PyMC", file=sys.stderr)
PY

# --- 3. the vLLM container ---------------------------------------------------
module load apptainer/1.3.6
apptainer --version

mkdir -p "$CONTAINER_DIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$HF_HOME"

if [[ -f "$VLLM_SIF" ]]; then
    echo
    echo "==> $VLLM_SIF already exists; skipping the container build"
else
    echo
    echo "==> building $VLLM_SIF from $VLLM_IMAGE (several minutes)"
    apptainer build "$VLLM_SIF" "$VLLM_IMAGE"
fi

# --- 4. the model weights ----------------------------------------------------
# Apptainer auto-binds only $HOME (§17.1), so without --bind the container
# cannot write to an HF_HOME on fastscratch and the download dies with
# "[Errno 30] Read-only file system".
echo
echo "==> downloading $MODEL into $HF_HOME (~61GB)"
apptainer exec --bind /mnt/fastscratch \
    --env HF_HOME="$HF_HOME" \
    "$VLLM_SIF" \
    hf download "$MODEL"

echo
echo "==> final fastscratch usage"
lfs quota -h -u "$USER" /mnt/fastscratch || true

cat <<EOF

Setup complete. Put these in $PWD/.env:

    LLM_BASE_URL=http://127.0.0.1:8000/v1
    LLM_MODEL=$MODEL
    LLM_CONTEXT_WINDOW=131072
    CODER_BASE_ENV=$BASE_ENV

Then submit from fastscratch (NOT home, data or localscratch):

    cd $FASTSCRATCH/coder-agent
    sbatch scripts/run_agent.sbatch /path/to/experiment_plan.json
EOF
