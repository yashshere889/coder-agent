#!/bin/bash -l
# Fallback when gpu-h100 is busy: the same run on pre-emptible A100 nodes.
#
#   sbatch scripts/run_agent_lowpri.sbatch /path/to/experiment_plan.json
#
# gpu-a100-lowbig (§14.6): gpu[06-09], 4x or 3x A100 80GB, 1-DAY limit, open to
# all users — but LOW PRIORITY. When the owning group (Cooper / DACDT / CS)
# submits, this job is pre-empted: killed and removed from the node (§14.2).
#
# Two consequences the main script does not have to think about:
#
#   --no-requeue is set. By default a pre-empted job is requeued (§9.4), which
#   here means restarting the whole agent run from scratch — regenerating code,
#   re-executing experiments, spending the model's time again on work that was
#   already done. Failing visibly is better than silently redoing hours of work.
#
#   --time is 24:00:00, the partition maximum. A run that needs longer belongs
#   on gpu-h100.
#
# A100 80GB fits the same model with the same TP=1 split, so nothing else
# changes.

#SBATCH --job-name=coder-agent-lowpri
#SBATCH --partition=gpu-a100-lowbig
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=coder_agent_lowpri_%j.log
#SBATCH --no-requeue
#SBATCH -D ./
#SBATCH --export=ALL

# gpu[06-07] have 4 A100s, gpu[08-09] have 3 — asking for 2 by count (not by
# type) schedules on whichever is free. Do not add --qos here: `low` is the
# partition QOS on lowpriority CPU partitions (§10), while these GPU partitions
# take the default, and setting it wrong makes the job pend rather than run.

# $0 inside a running job points at Slurm's spool copy of this script, not at
# the file you submitted, so dirname "$0" would resolve to the spool directory.
# $SLURM_SUBMIT_DIR is the directory sbatch was invoked from (§21, Table 2).
PROJECT_DIR="${SLURM_SUBMIT_DIR:-/mnt/fastscratch/users/$USER/coder-agent}"
exec /bin/bash -l "$PROJECT_DIR/scripts/run_agent.sbatch" "$@"
