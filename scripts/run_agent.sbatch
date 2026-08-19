#!/bin/bash
# Run the coder agent on one plan file, inside one Barkla job.
#
#   cd /mnt/fastscratch/users/$USER/coder-agent
#   sbatch scripts/run_agent.sbatch /path/to/experiment_plan.json
#
# Submit from fastscratch — Barkla's storage guidance says not to submit from
# home, data or localscratch.
#
# The model server and the agent share one allocation. The agent's own work is
# pure CPU (HTTP, subprocess management, AST checks), and the GPU partitions
# have 96-168 cores per node, so co-locating costs nothing and buys a localhost
# endpoint: no node-name discovery, no SSH tunnel, and nothing long-running on
# the login node, which Barkla forbids.

#SBATCH --job-name=coder-agent
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=coder_agent_%j.log

# Partition notes (Barkla GPU doc §14):
#   gpu-h100   4x H100 80GB   3-day limit   <- used here, 2 GPUs requested
#   gpu-l40s   2x L40S 48GB   3-day limit   <- needs TP=2 for this model; no GPU
#                                              left over for the experiment
#   gpu-v100   4x V100 16GB                 <- unusable: no BF16, too little VRAM
#
# Two GPUs are requested but the server is told to use only GPU 0. GPU 1 is
# handed to the generated experiment through CODER_EXPERIMENT_GPUS. Keep --time
# honest: gpu-h100 is shared and a shorter request schedules sooner.

set -euo pipefail

PLAN="${1:?usage: sbatch run_agent.sbatch <experiment_plan.json>}"

SCRATCH="/mnt/fastscratch/users/$USER"
PROJECT="${PROJECT:-$SCRATCH/coder-agent}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export VLLM_SIF="${VLLM_SIF:-$SCRATCH/containers/vllm.sif}"
export PATH="$HOME/.local/bin:$PATH"   # uv

cd "$PROJECT"

RUN_DIR="$SCRATCH/coder-agent-runs/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

# --- what the agent talks to -------------------------------------------------
export LLM_BASE_URL="http://127.0.0.1:8000/v1"
export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
# MUST match the --max-model-len the server actually starts with (MAXLEN below).
export LLM_CONTEXT_WINDOW="${MAXLEN:-131072}"

# --- what the generated experiments get --------------------------------------
export CODER_BASE_ENV="${CODER_BASE_ENV:-$SCRATCH/coder-agent-base}"
export CODER_EXPERIMENTS_DIR="$RUN_DIR/experiments"
# GPU 0 belongs to vLLM; the experiment gets GPU 1. Note this is an ordinal into
# the job's own allocation, not a node-wide device id.
export CODER_EXPERIMENT_GPUS="${CODER_EXPERIMENT_GPUS:-1}"
export CODER_MEMORY_LIMIT_GB="${CODER_MEMORY_LIMIT_GB:-64}"

echo "=== job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
echo "plan:        $PLAN"
echo "run dir:     $RUN_DIR"
echo "base env:    $CODER_BASE_ENV"
echo

# --- 1. the model server -----------------------------------------------------
export VLLM_LOG="$RUN_DIR/vllm.log"
export VLLM_PID_FILE="$RUN_DIR/vllm.pid"
TP="${TP:-1}" SERVER_GPUS=0 MAXLEN="${MAXLEN:-131072}" bash scripts/serve_vllm.sh

# Always take the server down: without this a failed agent leaves 61GB of
# weights resident and the job's remaining walltime burns with nothing running.
cleanup() {
    if [[ -f "$VLLM_PID_FILE" ]]; then
        echo "=== stopping the model server ==="
        kill "$(cat "$VLLM_PID_FILE")" 2>/dev/null || true
        sleep 5
        kill -9 "$(cat "$VLLM_PID_FILE")" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# --- 2. the agent ------------------------------------------------------------
uv sync --quiet
uv run coder-agent check || true    # informational; never blocks the run
echo
uv run coder-agent run "$PLAN" \
    ${CODER_DATA_DIR:+--data-dir "$CODER_DATA_DIR"} \
    --experiments-dir "$CODER_EXPERIMENTS_DIR"
STATUS=$?

echo
echo "=== finished with status $STATUS ==="
echo "results:  $CODER_EXPERIMENTS_DIR/*/results.json"
echo "reports:  $CODER_EXPERIMENTS_DIR/*/run_report.json"
echo "summary:  $CODER_EXPERIMENTS_DIR/summary.json"
exit $STATUS
