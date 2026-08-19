#!/usr/bin/env bash
# Start vLLM on this node and block until it answers.
#
# Called by run_agent.sbatch; runnable directly inside an interactive job:
#   srun -p gpu-h100 -N 1 --gres=gpu:h100:2 --pty /bin/bash    (§14.5)
#
#   TP=1 ./scripts/serve_vllm.sh                  # one GPU, one left for the experiment
#   TP=2 MAXLEN=262144 ./scripts/serve_vllm.sh    # both GPUs, full context
#
# TP=1 is the default deliberately. Qwen3-Coder-30B-A3B is ~61GB in BF16, which
# fits one H100 80GB at 0.90 utilisation with roughly 11GB left for KV cache
# (~110K tokens) — and leaves the node's second card entirely free for the
# generated experiment. An agent occupying both cards has nowhere to run the
# science.
set -uo pipefail

FASTSCRATCH="/mnt/fastscratch/users/$USER"
VLLM_SIF="${VLLM_SIF:-$FASTSCRATCH/containers/vllm.sif}"
MODEL="${LLM_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
MAXLEN="${MAXLEN:-131072}"
GPU_UTIL="${GPU_UTIL:-0.90}"
SERVER_GPUS="${SERVER_GPUS:-0}"
LOG="${VLLM_LOG:-vllm_${SLURM_JOB_ID:-local}.log}"
export HF_HOME="${HF_HOME:-$FASTSCRATCH/hf_cache}"

# Loading here as well as in the sbatch script, so this is runnable on its own
# in an interactive job. `module` is a shell function, absent in a non-login
# shell — hence `#!/bin/bash -l` on the sbatch side and this guard here.
if command -v module >/dev/null 2>&1; then
    module load apptainer/1.3.6 2>/dev/null || true
fi

if ! command -v apptainer >/dev/null 2>&1; then
    echo "error: apptainer not on PATH — run 'module load apptainer/1.3.6' first" >&2
    exit 1
fi
if [[ ! -f "$VLLM_SIF" ]]; then
    echo "error: $VLLM_SIF not found — run scripts/build_base_env.sh on barklaviz1 first" >&2
    exit 1
fi

echo "==> serving $MODEL on GPU(s) $SERVER_GPUS (TP=$TP), port $PORT, max-model-len $MAXLEN"
echo "==> server log: $LOG"

# --nv passes the host NVIDIA driver into the container; no driver is installed
# inside it (§17.1.3). Apptainer auto-binds only $HOME, so without --bind the
# container cannot read HF_HOME on fastscratch.
# No --tool-call-parser and no --reasoning-parser on purpose: this agent
# orchestrates in Python and reads delimited text sections, so it needs neither,
# and both are a version coupling that breaks quietly.
CUDA_VISIBLE_DEVICES="$SERVER_GPUS" apptainer exec --nv \
    --bind /mnt/fastscratch,/mnt/scratch,/tmp \
    --env HF_HOME="$HF_HOME" \
    "$VLLM_SIF" \
    vllm serve "$MODEL" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --tensor-parallel-size "$TP" \
        --max-model-len "$MAXLEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --max-num-seqs 4 \
    > "$LOG" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "${VLLM_PID_FILE:-vllm.pid}"

# Wait for /health rather than sleeping a fixed amount: weights load in anywhere
# between two and twenty minutes depending on page cache, and a fixed sleep is
# either wasteful or wrong.
echo "==> waiting for the server to come up"
for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "==> server is up after ${SECONDS}s"
        # The real KV budget is decided at load time and can be smaller than
        # --max-model-len asked for. Surface it: a prompt over the true ceiling
        # gets a 400 rather than a completion.
        grep -m1 "GPU KV cache size" "$LOG" || true
        exit 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "error: the server died during startup. Last 40 lines of $LOG:" >&2
        tail -40 "$LOG" >&2
        exit 1
    fi
    sleep 10
done

echo "error: the server did not answer /health within 40 minutes" >&2
tail -40 "$LOG" >&2
exit 1
