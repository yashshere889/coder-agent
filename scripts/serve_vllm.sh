#!/usr/bin/env bash
# Start vLLM on this node and block until it answers.
#
# Called by run_agent.sbatch; runnable directly for an interactive session.
#
#   TP=1 ./scripts/serve_vllm.sh     # one GPU, the other left for experiments
#   TP=2 MAXLEN=262144 ./scripts/serve_vllm.sh   # both GPUs, full context
#
# TP=1 is the default deliberately. Qwen3-Coder-30B-A3B is ~61GB in BF16, which
# fits one H100 at 0.90 utilisation with roughly 11GB left for KV cache (~110K
# tokens) — and leaves the node's second H100 entirely free for the generated
# experiment. An agent that occupies both cards has nowhere to run the science.
set -euo pipefail

SCRATCH="${SCRATCH:-/mnt/fastscratch/users/$USER}"
VLLM_SIF="${VLLM_SIF:-$SCRATCH/containers/vllm.sif}"
MODEL="${LLM_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
MAXLEN="${MAXLEN:-131072}"
GPU_UTIL="${GPU_UTIL:-0.90}"
SERVER_GPUS="${SERVER_GPUS:-0}"
LOG="${VLLM_LOG:-vllm_${SLURM_JOB_ID:-local}.log}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"

if [[ ! -f "$VLLM_SIF" ]]; then
    echo "error: $VLLM_SIF not found — run scripts/build_base_env.sh on a viz node first" >&2
    exit 1
fi

echo "==> serving $MODEL on GPU(s) $SERVER_GPUS (TP=$TP), port $PORT, max-model-len $MAXLEN"
echo "==> server log: $LOG"

# --nv passes the host NVIDIA driver in; without the explicit --bind the
# container cannot read HF_HOME on fastscratch.
# No --tool-call-parser and no --reasoning-parser on purpose: this agent
# orchestrates in Python and reads delimited text sections, so it depends on
# neither, and both are a version coupling that breaks quietly.
CUDA_VISIBLE_DEVICES="$SERVER_GPUS" apptainer exec --nv \
    --bind /mnt/fastscratch,/mnt/scratch \
    --env HF_HOME="$HF_HOME" \
    "$VLLM_SIF" \
    vllm serve "$MODEL" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --tensor-parallel-size "$TP" \
        --max-model-len "$MAXLEN" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --max-num-seqs 4 \
        --disable-log-requests \
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
        # --max-model-len asked for; surface it, because a prompt over the true
        # ceiling gets a 400 rather than a completion.
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
