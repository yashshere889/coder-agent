# coder-agent

Takes an experiment plan JSON, writes the code, runs it, diagnoses what breaks,
and repairs it — routing each failure to the thing that can actually fix it.

```
coder-agent run experiment_plan.json
```

Built for Barkla (2×H100), against `Qwen/Qwen3-Coder-30B-A3B-Instruct` served by
vLLM. Runs on a laptop too, against any OpenAI-compatible endpoint.

## Why it is built this way

Two failure modes from a previous system's run log shaped the whole design.

**1. All three fix attempts were spent regenerating code for
`ModuleNotFoundError: No module named 'pandas'`.** The code was never wrong; the
environment was. So this agent classifies a failure *before* deciding who
repairs it, and a missing package never reaches the code generator:

| Failure | Routed to | Costs a code attempt? |
|---|---|---|
| `ModuleNotFoundError` | install it, re-run the **unchanged** code | no |
| missing `.so` | stop — pip cannot fix it, say so plainly | no |
| `cannot import name X` | regenerate (code targets another version's API) | yes |
| missing data file / HTTP 404 | re-resolve the input, or declare a surrogate | no |
| OOM, CUDA OOM, timeout | halve the cost knobs **deterministically**, re-run | no |
| `SyntaxError` | regenerate (caught by `compile()`, never executed) | yes |
| any other traceback | regenerate, with the traceback quoted back | yes |
| exit 0 but no/empty/all-zero metrics | regenerate — a hollow success is still a failure | yes |
| unparseable model response | one targeted format repair, own budget | no |

Plus a **no-progress guard**: two identical failures escalate the next request
from "fix this" to "solve it a different way"; a third stops the run. Three
identical failures is the exact shape of the run this replaces.

**2. The generated experiment expected "placeholder CSV files in `data/`"** that
nothing had ever created — and would have produced posterior means and a
supported/refuted verdict off invented numbers. So every input resolves to a
declared kind (`real_local`, `real_download`, `synthetic_surrogate`), and if any
input is a surrogate then `hypothesis_outcome` is forced to `not_assessable`.
That rule is enforced in Python, not requested in a prompt. The metrics are
still reported; the claim about the world is not.

## Quickstart (laptop)

```bash
uv sync --extra dev
uv run pytest                       # 71 tests, no network, no model, no cluster
uv run coder-agent check            # what is and isn't set up here
```

With a model endpoint running:

```bash
cp .env.example .env                # point LLM_BASE_URL at it
uv run coder-agent run experiment_plan.json --data-dir ~/staged-data
```

## Barkla

One-time, **on a viz node** (`barklaviz1.liv.ac.uk` — long builds are killed on
the login node):

```bash
cd /mnt/fastscratch/users/$USER/coder-agent
bash scripts/build_base_env.sh
```

That builds the vLLM Apptainer image, the pinned base environment every
experiment inherits, and downloads the model (~61GB) into `HF_HOME` on
fastscratch. It has to happen ahead of time because compute nodes frequently
have **no outbound network** — which is also why a package missing at run time
reports "add it to `build_base_env.sh`" rather than retrying an unreachable
index.

Then, per run:

```bash
sbatch scripts/run_agent.sbatch /path/to/experiment_plan.json
```

### GPU split

`run_agent.sbatch` requests 2 H100s and gives the model server **one** of them
(`TP=1`, ~61GB of BF16 weights at 0.90 utilisation, ~110K tokens of KV cache).
GPU 1 goes to the generated experiment via `CODER_EXPERIMENT_GPUS=1`. An agent
that occupies both cards has nowhere to run the science.

Need the full 256K context and no GPU experiment? `TP=2 MAXLEN=262144 sbatch ...`
— and set `LLM_CONTEXT_WINDOW` to match, since a prompt over the server's real
ceiling gets a 400 rather than a completion. The server prints its true KV
budget as `GPU KV cache size: N tokens` at startup; `serve_vllm.sh` greps it out
for you.

## What a run produces

```
experiments/H1/
├── run.py                  # the version that ran (or last was tried)
├── results.json            # metrics, written by the fixed scaffold
├── run_report.json         # status, every attempt, every repair, budgets used
├── data_provenance.json    # per input: real or surrogate, and why
├── environment.json        # the overlay venv and everything installed into it
├── logs/{stdout,stderr}.log
└── attempts/attempt_N/     # each failed version + its diagnosis + evidence
```

`run_report.json` is the one to read: it shows not just that an experiment
failed, but which *kind* of failure it was and what the harness did about it.

## Architecture

```
plan.py      parse/validate the plan JSON — the entire input contract
data.py      resolve inputs; the provenance gate
envman.py    base env probe, per-experiment uv overlay, import → package
codegen.py   splice model sections into templates/run.py.template
executor.py  rlimit'd subprocess, timeout, env allowlist, safety scan
diagnose.py  ★ classify the failure (pure function, no LLM)
repair.py    ★ route it; deterministic downscale; no-progress guard
loop.py      the state machine
```

Design choices worth knowing before changing anything:

- **The model never writes the whole file.** It fills named sections spliced into
  a fixed scaffold that owns the metadata header and the orchestration footer, so
  the guarantee that `results.json` gets written survives any bad generation.
- **Splicing is `str.replace()` on `__TOKEN__`, not `str.format()`.** Generated
  Python is full of literal braces; format substitution fails on correct code.
- **Code-bearing responses use delimited sections, never JSON.** Putting Python
  inside a JSON string forces the model to hand-escape multi-line code on every
  response, which is the most reliable way to break a code-generating model.
- **No tool-calling.** The loop is orchestrated in Python and the model emits
  text. That drops a dependency on the server's tool-call parser matching the
  model's chat template — a coupling that breaks silently and costs a run.
- **`diagnose.py` reads only text.** No filesystem, no network, no model. That is
  what makes it testable against a corpus of real tracebacks, which is where its
  correctness actually comes from.
- **`executor.py` is a bounds check, not a security boundary.** No Docker daemon
  exists on Barkla. The trust model is our own model responding to our own plan.

## Configuration

Everything is env vars (see `.env.example`); `config.py` is the only module that
reads them.

| | |
|---|---|
| `LLM_BASE_URL` / `LLM_MODEL` | the OpenAI-compatible endpoint |
| `LLM_CONTEXT_WINDOW` | must match the server's `--max-model-len` |
| `CODER_BASE_ENV` | the pre-baked env experiments inherit |
| `CODER_EXPERIMENT_GPUS` | GPU ordinals the experiment may use (`""` = CPU-only) |
| `CODER_MAX_CODE_ATTEMPTS` | 4 — genuine code defects |
| `CODER_MAX_ENV_REPAIRS` | 6 — installs |
| `CODER_MAX_FORMAT_RETRIES` | 2 — unparseable responses |
| `CODER_TIMEOUT_{LOW,MEDIUM,HIGH}` | wall clock, by the plan's own complexity |

Three separate budgets because they fail for unrelated reasons, and one draining
the others is how the previous system ran out of attempts without ever having a
code problem.

## Testing

```bash
uv run pytest                                   # everything
uv run pytest tests/test_diagnose.py -q         # the classifier corpus
uv run pytest tests/test_loop.py -q             # real venvs, real subprocesses
```

No test reaches a model, a cluster, or the network. `tests/test_loop.py`'s first
test is the regression for the observed failure: a missing package must cost
exactly one install and **zero** regenerations.

Debug the classifier against a real log without running anything:

```bash
uv run coder-agent classify path/to/job.log
```
