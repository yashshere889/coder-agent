#!/usr/bin/env bash
# Pick whichever GPU partition can start you RIGHT NOW, and submit there.
#
#   ./scripts/submit.sh /mnt/fastscratch/users/$USER/experiment_plan.json
#
# Waiting behind a queue costs more wall-clock than a slower card does. This
# asks sinfo which partitions have idle nodes, picks the best one that can start
# immediately, and works out the GPU count, GPU type and tensor-parallel size
# that partition needs — those differ per partition and getting them wrong fails
# in ways that are slow to notice:
#
#   * gpu-a-lowsmall mixes A100 80GB with A40 48GB. 57GiB of weights do not fit
#     an A40 at TP=1, so this only ever targets its A100 nodes explicitly.
#   * The L40S partitions are 48GB per card, so the model must shard across two
#     (TP=2) and the experiment runs CPU-only.
#   * The 80GB partitions need one card, leaving the second free for the
#     experiment — but only if two were requested.
#
# Everything is checked before submission, because a job that fails on a missing
# file still costs a queue slot and a log to read.
set -uo pipefail

PLAN="${1:?usage: ./scripts/submit.sh <experiment_plan.json> [extra sbatch args...]}"
shift || true

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"

WALLTIME="${WALLTIME:-6:00:00}"

say()  { printf '%s\n' "$*"; }
fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight --
say "==> preflight"

[[ -f "$PLAN" ]] || fail "plan file not found: $PLAN"
PLAN="$(cd "$(dirname "$PLAN")" && pwd)/$(basename "$PLAN")"

python3 - "$PLAN" <<'PY' || fail "the plan file is not usable (see above)"
import json, sys
try:
    doc = json.load(open(sys.argv[1]))
except Exception as exc:
    sys.exit(f"  not valid JSON: {exc}")
plans = doc.get("experiment_plans")
if not isinstance(plans, list) or not plans:
    sys.exit("  'experiment_plans' is missing or empty")
runnable = [p for p in plans if p.get("feasible", True)]
if not runnable:
    sys.exit("  every plan is marked feasible=false; nothing would run")
print(f"  plan ok: {len(runnable)} runnable — {[p.get('hypothesis_id') for p in runnable]}")
PY

case "$PWD" in
    /mnt/scratch/*|/mnt/fastscratch/*) ;;
    *) fail "submit from scratch or fastscratch — compute nodes cannot write elsewhere (Barkla §8). Currently in $PWD" ;;
esac

BASE_ENV="${CODER_BASE_ENV:-/mnt/fastscratch/users/$USER/coder-agent-base}"
if [[ -x "$BASE_ENV/bin/python" ]]; then
    _n=$("$BASE_ENV/bin/python" - <<'PY'
mods = ["numpy","pandas","scipy","sklearn","statsmodels","matplotlib","pymc","arviz","geopandas","cmdstanpy"]
import importlib.util
print(sum(1 for m in mods if importlib.util.find_spec(m)))
PY
)
    say "  base env: $_n/10 key packages present"
    [[ "${_n:-0}" -lt 5 ]] && say "  WARNING: base env looks incomplete — run scripts/build_base_env.sh on barklaviz1"
else
    say "  WARNING: no base env at $BASE_ENV — experiments will install everything themselves"
fi

command -v sinfo >/dev/null 2>&1 || fail "sinfo not found — run this on a Barkla login node"

# ------------------------------------------------------------ partition scan --
# Fields: partition | gres request | TP | script | description
CANDIDATES=(
    "gpu-h100|gpu:h100:2|1|run_agent.sbatch|H100 80GB, common, 3-day"
    "gpu-a-lowsmall|gpu:a100:1|1|run_agent_lowpri.sbatch|A100 80GB, pre-emptible, 1-day"
    "gpu-a100-lowbig|gpu:1|1|run_agent_lowpri.sbatch|A100 80GB, pre-emptible, 1-day"
    "gpu-l40s|gpu:2|2|run_agent.sbatch|2x L40S 48GB (TP=2), common, 3-day"
    "gpu-l40s-low|gpu:2|2|run_agent_lowpri.sbatch|2x L40S 48GB (TP=2), pre-emptible, 1-day"
)

# Count nodes in a partition that are idle. For gpu-a-lowsmall the node's own
# gres is inspected too, since an idle A40 there is no use to us.
idle_nodes() {
    local partition="$1" want_type="$2" count=0 line state gres
    while read -r line; do
        state="$(awk '{print $2}' <<<"$line")"
        gres="$(awk '{print $3}' <<<"$line")"
        [[ "$state" == idle* ]] || continue
        if [[ -n "$want_type" && "$gres" != *"$want_type"* ]]; then continue; fi
        count=$((count + 1))
    done < <(sinfo -h -N -p "$partition" -o "%N %t %G" 2>/dev/null)
    printf '%s' "$count"
}

say ""
say "==> scanning partitions for a node that can start now"

CHOSEN=""
for entry in "${CANDIDATES[@]}"; do
    IFS='|' read -r partition gres tp script description <<<"$entry"
    want_type=""
    [[ "$gres" == *a100* ]] && want_type="a100"
    n="$(idle_nodes "$partition" "$want_type")"
    printf '  %-18s %s idle   %s\n' "$partition" "${n:-0}" "$description"
    if [[ -z "$CHOSEN" && "${n:-0}" -gt 0 ]]; then
        CHOSEN="$entry"
    fi
done

if [[ -z "$CHOSEN" ]]; then
    say ""
    say "  no partition has a fully idle node; falling back to the shortest queue"
    best=999
    for entry in "${CANDIDATES[@]}"; do
        IFS='|' read -r partition _ _ _ _ <<<"$entry"
        depth="$(squeue -h -p "$partition" -t PENDING 2>/dev/null | wc -l)"
        if [[ "${depth:-999}" -lt "$best" ]]; then best="$depth"; CHOSEN="$entry"; fi
    done
fi

IFS='|' read -r PARTITION GRES TP SCRIPT DESCRIPTION <<<"$CHOSEN"

# ------------------------------------------------------------------- submit --
say ""
say "==> submitting to $PARTITION ($DESCRIPTION)"
say "    gres=$GRES  TP=$TP  time=$WALLTIME  script=$SCRIPT"

OUT="$(sbatch -p "$PARTITION" --gres="$GRES" --time="$WALLTIME" \
        --export="ALL,TP=$TP" "$@" "scripts/$SCRIPT" "$PLAN" 2>&1)"
status=$?
say "    $OUT"
[[ $status -eq 0 ]] || fail "submission refused (see above)"

JOBID="$(grep -oE '[0-9]+' <<<"$OUT" | tail -1)"
say ""
say "watch it:"
say "  squeue -u \$USER"
say "  tail -F $PROJECT/*_${JOBID}.log"
say "  tail -F /mnt/fastscratch/users/\$USER/coder-agent-runs/${JOBID}/vllm.log"
