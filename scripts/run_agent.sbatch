#!/bin/bash -l
# Run the coder agent on one plan file, inside one Barkla job.
#
#   cd /mnt/fastscratch/users/$USER/coder-agent
#   sbatch scripts/run_agent.sbatch /path/to/experiment_plan.json
#
# The `-l` on the shebang is REQUIRED (Barkla docs §8): it requests a login
# shell so `module load` works inside the script. Without it every module
# command below silently fails.
#
# Submit from scratch or fastscratch. Compute nodes have write access ONLY to
# those two filesystems — home and data are read-only from a node, and
# localscratch differs per node (§5.1, §8).

#SBATCH --job-name=coder-agent
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --time=12:00:00
#SBATCH --output=coder_agent_%j.log
#SBATCH -D ./
#SBATCH --export=ALL

# --- why these directives, and not others -----------------------------------
#
# gpu-h100 (§14.1.4): gpu[31-33], each 4x H100 SXM 80GB HBM3, 96 Intel Xeon
#   Platinum 8468 cores, 2048GB RAM, 7.68TB local NVMe. Common partition — all
#   users. Time limit 3 days.
#
# --gres=gpu:h100:2 requests two of the node's four cards by type (§14.4).
#   GPU 0 runs vLLM; GPU 1 is handed to the generated experiment.
#
# There is deliberately NO --cpus-per-task, no --ntasks and no --mem. Barkla's
#   GPU guidance (§14.10) is explicit: "we recommend not manually requesting the
#   number of CPU cores or memory per core (or total memory), as they are
#   automatically allocated based on the number of GPUs requested". §8 says the
#   same for memory on every partition. Requesting them here causes pending
#   jobs or oversubscription, not faster ones.
#
# --time is set explicitly because the *default* is 8 hours (§8), which will cut
#   a long agent run off mid-experiment. Keep it honest anyway: gpu-h100 is a
#   3-node common partition and a shorter request schedules sooner.
#
# Fallback when gpu-h100 is busy: run_agent_lowpri.sbatch (A100 80GB,
#   pre-emptible, 1-day limit).

set -uo pipefail

PLAN="${1:?usage: sbatch run_agent.sbatch <experiment_plan.json>}"

# Resolve the plan to an absolute path BEFORE the `cd "$PROJECT"` below, and
# check it exists BEFORE spending three minutes of GPU time loading 61GB of
# weights for a run that cannot start. A relative path resolved after the cd
# silently pointed somewhere else, and the failure only surfaced once the server
# was already up — the agent exited immediately and the trap tore the server
# straight back down, which reads like a server problem and is not one.
if [[ ! -f "$PLAN" ]]; then
    echo "error: plan file not found: $PLAN" >&2
    echo "       (resolved from $(pwd); pass an absolute path)" >&2
    exit 2
fi
PLAN="$(cd "$(dirname "$PLAN")" && pwd)/$(basename "$PLAN")"

if ! python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$PLAN" 2>/dev/null; then
    echo "error: $PLAN is not valid JSON" >&2
    exit 2
fi

# --- modules -----------------------------------------------------------------
# `module purge` first (§7.1): loading inside the job script rather than
# inheriting the submitting shell's modules is what makes a run reproducible.
module purge
module load apptainer/1.3.6

# --- filesystems (§5.1) -------------------------------------------------------
FASTSCRATCH="/mnt/fastscratch/users/$USER"    # 500GB / 500k files, no backup
LOCALSCRATCH="/tmp/users/$USER"               # node-local NVMe, no quota, purged after 30d
PROJECT="${PROJECT:-$FASTSCRATCH/coder-agent}"

export HF_HOME="${HF_HOME:-$FASTSCRATCH/hf_cache}"
export VLLM_SIF="${VLLM_SIF:-$FASTSCRATCH/containers/vllm.sif}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$FASTSCRATCH/apptainer_cache}"
export PATH="$HOME/.local/bin:$PATH"          # uv

cd "$PROJECT" || { echo "error: $PROJECT not found" >&2; exit 1; }

RUN_DIR="$FASTSCRATCH/coder-agent-runs/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR" "$LOCALSCRATCH"

# --- threading (§8) -----------------------------------------------------------
# Without this, any OpenMP-linked library in a generated experiment (numpy,
# scipy, sklearn, Stan) takes ALL 96 cores on the node as its default and
# overloads it. Half the cores to the experiment is generous and safe; vLLM's
# own threads and the agent's Python are cheap alongside it.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

# --- what the agent talks to ---------------------------------------------------
export LLM_BASE_URL="http://127.0.0.1:8000/v1"
export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
export MAXLEN="${MAXLEN:-131072}"
# MUST match the --max-model-len the server actually starts with, or a long fix
# prompt gets a 400 back instead of a completion.
export LLM_CONTEXT_WINDOW="$MAXLEN"

# --- what the generated experiments get ----------------------------------------
export CODER_BASE_ENV="${CODER_BASE_ENV:-$FASTSCRATCH/coder-agent-base}"
export CODER_EXPERIMENTS_DIR="$RUN_DIR/experiments"
# Overlay venvs go on localscratch, NOT fastscratch. §5.1.7: workloads with many
# small files should use localscratch to avoid inode quota violations, and a venv
# is thousands of small files. It is rebuilt per job, so nothing is lost.
export CODER_VENV_ROOT="$LOCALSCRATCH/coder-agent-venvs-${SLURM_JOB_ID}"
# GPU 0 belongs to vLLM; the experiment gets GPU 1. Ordinal into this job's own
# allocation, not a node-wide device id.
export CODER_EXPERIMENT_GPUS="${CODER_EXPERIMENT_GPUS:-1}"
export CODER_MEMORY_LIMIT_GB="${CODER_MEMORY_LIMIT_GB:-256}"

echo "=== job $SLURM_JOB_ID on $(hostname) ==="
echo "partition:   $SLURM_JOB_PARTITION"
echo "plan:        $PLAN"
echo "project:     $PROJECT"
echo "run dir:     $RUN_DIR"
echo "base env:    $CODER_BASE_ENV"
echo "venv root:   $CODER_VENV_ROOT  (localscratch — no inode quota)"
echo "started:     $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo

# --- 1. the model server -------------------------------------------------------
export VLLM_LOG="$RUN_DIR/vllm.log"
export VLLM_PID_FILE="$RUN_DIR/vllm.pid"
TP="${TP:-1}" SERVER_GPUS=0 bash scripts/serve_vllm.sh || {
    echo "error: the model server never came up; see $VLLM_LOG" >&2
    exit 1
}

# Always take the server down. Without this a failed agent leaves 61GB of
# weights resident on a shared partition for the rest of the walltime.
cleanup() {
    if [[ -f "$VLLM_PID_FILE" ]]; then
        echo "=== stopping the model server ==="
        kill "$(cat "$VLLM_PID_FILE")" 2>/dev/null || true
        sleep 10
        kill -9 "$(cat "$VLLM_PID_FILE")" 2>/dev/null || true
    fi
    # localscratch is purged automatically after 30 days, but a long batch of
    # jobs on one node would otherwise pile up venvs in the meantime.
    rm -rf "$CODER_VENV_ROOT"
}
trap cleanup EXIT

# --- 2. the agent ---------------------------------------------------------------
uv sync --quiet
uv run coder-agent check || true      # informational; never blocks the run
echo
uv run coder-agent run "$PLAN" \
    ${CODER_DATA_DIR:+--data-dir "$CODER_DATA_DIR"} \
    --experiments-dir "$CODER_EXPERIMENTS_DIR"
STATUS=$?

echo
echo "=== finished with status $STATUS at $(date) ==="
echo "results:  $CODER_EXPERIMENTS_DIR/*/results.json"
echo "reports:  $CODER_EXPERIMENTS_DIR/*/run_report.json"
echo "summary:  $CODER_EXPERIMENTS_DIR/summary.json"
exit $STATUS
