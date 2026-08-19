# Running on Barkla

Every command you need, in order. Section numbers refer to the Research IT
technical docs, so you can check any claim here against the source.

## 0. Connect

```bash
ssh <user>@barklalogin1.liv.ac.uk
```

On campus or on the VPN only. Three node types, and using the wrong one is the
most common way to lose an afternoon (§4):

| Node | For | Not for |
|---|---|---|
| `barklalogin1` | editing, submitting jobs | builds, downloads, anything long — **killed without warning** |
| `barklaviz1` / `barklaviz2` | lengthy builds, big downloads, GPU debugging (≤8 cores, ≤8 h) | production runs |
| compute nodes (via Slurm) | everything real | direct login |

## 1. Check your storage before you start

```bash
/opt/apps/user_tools/display_disk_usage_quota_bars.sh
```

```bash
lfs quota -h -u $USER /mnt/fastscratch
```

Setup needs ~75GB on `fastscratch`, whose quota is **500GB / 500k files** (§5.1).
Worth knowing which filesystem does what:

| Path | Size / inodes | Backed up | Use |
|---|---|---|---|
| `/users/$USER` (home) | 75GB / 100k | yes | scripts only — **never** a Python env, never submit from here |
| `/mnt/data1/users/$USER` | 2.5TB / 300k | yes | datasets; **read-only on compute nodes** |
| `/mnt/scratch/users/$USER` | 2.0TB / 300k | no | work dir, submit from here |
| `/mnt/fastscratch/users/$USER` | 500GB / 500k | no | work dir, fast, submit from here — used by this project |
| `/tmp/users/$USER` (localscratch) | no quota, purged after 30 d | no | node-local; overlay venvs go here |

**Jobs must be submitted from `scratch` or `fastscratch`** — compute nodes have
write access only to those two (§8).

## 2. Get the code there

```bash
mkdir -p /mnt/fastscratch/users/$USER && cd /mnt/fastscratch/users/$USER
```

```bash
git clone https://github.com/yashshere889/coder-agent.git && cd coder-agent
```

## 3. One-time setup — on a viz node

```bash
ssh <user>@barklaviz1.liv.ac.uk
```

```bash
cd /mnt/fastscratch/users/$USER/coder-agent && bash scripts/build_base_env.sh
```

Builds the vLLM `.sif`, the pinned base environment, and downloads the model
(~61GB). It refuses to run on the login node. Takes 30–60 minutes, mostly the
download.

Then write `.env`:

```bash
cat > .env <<'EOF'
LLM_BASE_URL=http://127.0.0.1:8000/v1
LLM_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct
LLM_CONTEXT_WINDOW=131072
CODER_BASE_ENV=/mnt/fastscratch/users/USERNAME/coder-agent-base
CODER_MAX_CODE_ATTEMPTS=4
CODER_MAX_ENV_REPAIRS=6
CODER_TIMEOUT_MEDIUM=1800
CODER_TIMEOUT_HIGH=7200
EOF
sed -i "s/USERNAME/$USER/" .env
```

Verify before you queue anything:

```bash
cd /mnt/fastscratch/users/$USER/coder-agent && uv run coder-agent check
```

## 4. Submit

```bash
cd /mnt/fastscratch/users/$USER/coder-agent
```

```bash
sbatch scripts/run_agent.sbatch /mnt/fastscratch/users/$USER/plans/experiment_plan.json
```

With data you have staged yourself (CMS extracts under a DUA, say):

```bash
CODER_DATA_DIR=/mnt/data1/users/$USER/cms sbatch scripts/run_agent.sbatch plan.json
```

When `gpu-h100` is busy, the pre-emptible A100 fallback:

```bash
sbatch scripts/run_agent_lowpri.sbatch plan.json
```

## 5. Monitor

```bash
squeue -u $USER
```

```bash
tail -f coder_agent_<jobid>.log
```

```bash
tail -f /mnt/fastscratch/users/$USER/coder-agent-runs/<jobid>/vllm.log
```

States: `PD` pending, `CF` configuring, `R` running, `CG` completing, `F` failed
(§8). Why a job is pending:

```bash
squeue -u $USER -o "%.10i %.12P %.10T %.10M %.6D %R"
```

`QOSMaxCpuPerUserLimit` means you have hit the 400-core cap across all your
running jobs (§10) — wait, don't resubmit.

What the partition looks like right now:

```bash
sinfo -p gpu-h100 -o "%.12P %.6a %.14l %.6D %.6t %N"
```

Cancel:

```bash
scancel <jobid>
```

After it finishes:

```bash
sacct -j <jobid> --format=JobID,JobName,Partition,Elapsed,State,ExitCode,MaxRSS
```

## 6. Read the results

```bash
cd /mnt/fastscratch/users/$USER/coder-agent-runs/<jobid>/experiments
```

```bash
cat summary.json | python3 -m json.tool | head -40
```

```bash
cat H1/run_report.json | python3 -m json.tool
```

`run_report.json` is the one to read — it shows not just whether an experiment
worked, but which *kind* of failure each attempt hit and what the harness did
about it. `data_provenance.json` next to it says which inputs were real.

## 7. Interactive debugging

```bash
srun -p gpu-h100 -N 1 --gres=gpu:h100:2 --time=2:00:00 --pty /bin/bash
```

Then, on the node:

```bash
module purge && module load apptainer/1.3.6
```

```bash
cd /mnt/fastscratch/users/$USER/coder-agent && TP=1 bash scripts/serve_vllm.sh
```

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

## Why the job scripts look the way they do

Each of these is a documented Barkla rule, not a preference.

- **`#!/bin/bash -l`** — required, or `module load` silently does nothing (§8).
- **`module purge` before loading** — so the job doesn't inherit whatever your
  submitting shell had loaded (§7.1).
- **No `--cpus-per-task`, no `--mem`** — on GPU partitions cores and memory are
  allocated automatically from the GPU count, and requesting them manually causes
  pending or oversubscribed jobs (§14.10, §8).
- **`--time` set explicitly** — the default is 8 hours, which will cut a long run
  off mid-experiment (§8).
- **`OMP_NUM_THREADS` set explicitly** — otherwise any OpenMP-linked library in a
  generated experiment takes all 96 cores on the node and overloads it (§8).
- **Overlay venvs on localscratch** — a venv is thousands of small files, and
  `fastscratch` has a 500k inode quota; §5.1.7 says to put many-small-file
  workloads on localscratch. The venvs are rebuilt per job, so nothing is lost.
- **The base env is inherited, not copied** — `--system-site-packages`, exactly as
  §16.11.3.4 recommends, because duplicated packages across environments "often
  lead to exceeding the file number quota".
- **`--no-requeue` on the low-priority script** — a pre-empted job is requeued by
  default (§9.4), which for this agent means silently redoing hours of finished
  work.

## Hardware

`gpu-h100` — `gpu[31-33]`, common partition, all users, **3-day limit** (§14.1.4):

- 4 × H100 SXM 80GB HBM3 per node
- 96 Intel Xeon Platinum 8468 cores, 2048 GB RAM
- 7.68 TB local NVMe

The job asks for 2 of the 4 cards. vLLM gets GPU 0 (~61GB of BF16 weights at
0.90 utilisation, ~110K tokens of KV cache); the generated experiment gets GPU 1.

Fallbacks, in order of preference:

| Partition | GPUs | Limit | Note |
|---|---|---|---|
| `gpu-h100` | 4 × H100 80GB | 3 days | first choice |
| `gpu-a100-lowbig` | 4/3 × A100 80GB | 1 day | pre-emptible |
| `gpu-l40s` | 2 × L40S 48GB | 3 days | needs `TP=2`; no GPU left for the experiment |
| `gpu-v100` | 4 × V100 16GB | 3 days | unusable — no BF16, too little VRAM |
