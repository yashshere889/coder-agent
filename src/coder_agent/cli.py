"""Command-line entry point.

    coder-agent run plan.json                 # the whole thing
    coder-agent check                         # is this node set up to run one?
    coder-agent classify job.log              # what would the classifier say?

`check` exists because every expensive failure mode of this agent is a setup
problem — a model server that is not up, a base env that was never built, a
compute node with no network — and each takes seconds to detect here versus an
hour of allocation to discover mid-run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import diagnose, envman
from .config import settings
from .llm import get_chat_model
from .loop import run_all
from .plan import PlanError, load_plans


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        stream=sys.stderr,
    )
    # The HTTP client logs every request at INFO, which drowns the agent's own
    # progress lines in a long run.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def cmd_run(args: argparse.Namespace) -> int:
    try:
        plans = load_plans(args.plan, hypothesis_ids=args.hypothesis or None)
    except PlanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    model = get_chat_model()
    ok, detail = model.health()
    if not ok:
        print(f"error: {detail}", file=sys.stderr)
        print(
            "hint: start the server with scripts/serve_vllm.sh, or point LLM_BASE_URL "
            "at one that is already running.",
            file=sys.stderr,
        )
        return 3

    print(f"model: {detail}")
    print(f"plans: {', '.join(p.hypothesis_id for p in plans)}")
    print(f"runtime: {json.dumps(envman.describe())}")

    summary = run_all(
        plans,
        model,
        staging_dir=Path(args.data_dir) if args.data_dir else None,
        experiments_root=Path(args.experiments_dir) if args.experiments_dir else None,
    )

    print()
    for experiment in summary["experiments"]:
        evaluation = experiment.get("evaluation") or {}
        print(f"  {experiment['hypothesis_id']}: {experiment['status']}")
        print(f"    outcome:  {evaluation.get('hypothesis_outcome', 'n/a')}")
        print(f"    validity: {evaluation.get('methodological_validity', 'n/a')}")
        if experiment.get("reason"):
            print(f"    reason:   {experiment['reason']}")
        print(f"    report:   {experiment['experiment_dir']}/run_report.json")

    print(f"\n{summary['completed']}/{summary['total']} experiments completed")
    return 0 if summary["completed"] == summary["total"] else 1


def cmd_check(args: argparse.Namespace) -> int:
    """Report everything that has to be true before a run is worth starting."""
    runtime = envman.describe()
    print("runtime")
    for key, value in runtime.items():
        print(f"  {key:22s} {value}")

    print("\nsettings")
    for key in ("llm_base_url", "llm_model", "llm_context_window", "base_env", "experiment_gpus"):
        print(f"  {key:22s} {getattr(settings, key) or '(unset)'}")

    problems = []
    if not runtime["uv"]:
        problems.append("uv is not on PATH — venv creation falls back to the slower stdlib path")
    if not settings.base_env:
        problems.append(
            "CODER_BASE_ENV is unset — every experiment starts from a bare venv and will "
            "install the whole scientific stack itself (run scripts/build_base_env.sh)"
        )
    elif not Path(settings.base_env).exists():
        problems.append(f"CODER_BASE_ENV points at {settings.base_env}, which does not exist")
    if not runtime["network"]:
        problems.append(
            "no outbound network from this node — anything not already in the base env "
            "cannot be installed during a run"
        )

    print("\nmodel server")
    ok, detail = get_chat_model().health()
    print(f"  {'ok' if ok else 'FAIL'}  {detail}")
    if not ok:
        problems.append(detail)

    if problems:
        print("\nproblems")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("\nno problems found")
    return 0 if ok else 1


def cmd_classify(args: argparse.Namespace) -> int:
    """Run the classifier over a saved log — for debugging the router itself."""
    text = Path(args.logfile).read_text(errors="replace")
    diagnosis = diagnose.classify(
        exit_code=args.exit_code, stdout="", stderr=text, timed_out=args.timed_out
    )
    if diagnosis is None:
        print("no failure detected in this log")
        return 0
    print(json.dumps(diagnosis.to_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coder-agent", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="generate, execute and repair the experiments in a plan file")
    run.add_argument("plan", help="path to the experiment plan JSON")
    run.add_argument(
        "--hypothesis", action="append", help="only run this hypothesis id (repeatable)"
    )
    run.add_argument("--data-dir", help="directory of data files you have already staged")
    run.add_argument("--experiments-dir", help="override CODER_EXPERIMENTS_DIR")
    run.set_defaults(func=cmd_run)

    check = sub.add_parser("check", help="report whether this node can run an experiment")
    check.set_defaults(func=cmd_check)

    classify = sub.add_parser("classify", help="diagnose a saved log with the error classifier")
    classify.add_argument("logfile")
    classify.add_argument("--exit-code", type=int, default=1)
    classify.add_argument("--timed-out", action="store_true")
    classify.set_defaults(func=cmd_classify)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
