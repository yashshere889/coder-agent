"""The agent loop: plan in, executed experiment and an honest report out.

An explicit state machine, deliberately not a graph framework. The whole thing
is one readable function plus its helpers, because when this fails at 3am inside
a SLURM job the thing that matters is being able to read the control flow in one
pass.

    resolve data -> provision env -> generate -> preflight -> execute
                                        ^                       |
                                        |                   classify
                                        |                       |
                                        +------ repair <--------+

The repair edge is the whole design. It does NOT always lead back to the code
generator: an install, a data re-resolution and a downscale all re-run the
*existing* code, and only a genuine code defect spends a code attempt. Three
separate budgets are tracked for the same reason — they fail for unrelated
reasons and should not drain each other.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import codegen, data, diagnose, envman, executor, repair
from .config import Settings, settings as default_settings
from .diagnose import Diagnosis
from .llm import ChatModel
from .plan import ExperimentPlan

logger = logging.getLogger(__name__)

STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_BUDGET_EXHAUSTED = "budget_exhausted"
STATUS_TERMINAL = "stopped_unfixable"
# Distinct from the above on purpose: "the same failure kept recurring" is a
# different thing for a human to act on than "this cannot be fixed from here",
# and collapsing them hides which one happened.
STATUS_NO_PROGRESS = "stopped_no_progress"


@dataclass
class Attempt:
    """One pass through preflight+execute, and what the harness did about it."""

    number: int
    stage: str
    diagnosis: dict[str, Any] | None = None
    repair: dict[str, Any] | None = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.number,
            "stage": self.stage,
            "diagnosis": self.diagnosis,
            "repair": self.repair,
            "duration_seconds": round(self.duration_seconds, 2),
        }


@dataclass
class ExperimentReport:
    """Everything a human needs to judge the run without re-reading the logs."""

    hypothesis_id: str
    status: str
    experiment_dir: str
    reason: str = ""
    results: dict[str, Any] | None = None
    evaluation: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    env: dict[str, Any] = field(default_factory=dict)
    code_attempts_used: int = 0
    env_repairs_used: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "status": self.status,
            "reason": self.reason,
            "experiment_dir": self.experiment_dir,
            "evaluation": self.evaluation,
            "results": self.results,
            "data_provenance": self.provenance,
            "assumptions_made": self.assumptions,
            "attempts": self.attempts,
            "environment": self.env,
            "budgets": {
                "code_attempts_used": self.code_attempts_used,
                "env_repairs_used": self.env_repairs_used,
            },
            "duration_seconds": round(self.duration_seconds, 2),
        }


def _read_results(path: Path) -> tuple[dict[str, Any] | None, bool]:
    """Read the experiment's results.json. Returns (parsed, present)."""
    if not path.exists():
        return None, False
    try:
        return json.loads(path.read_text()), True
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("results.json exists but could not be read: %s", exc)
        return None, True


def _preflight(code: str) -> Diagnosis | None:
    """Every check that can be made without running anything.

    Ordered cheapest-first, and all of them before a subprocess exists: a
    SyntaxError found by `compile()` costs microseconds, while finding the same
    thing by launching the script costs a process, a venv activation and, on a
    GPU experiment, a CUDA context.
    """
    syntax_error = executor.compile_check(code)
    if syntax_error:
        return diagnose.syntax_failure(syntax_error)

    findings = executor.static_safety_check(code)
    if findings:
        return diagnose.safety_failure(findings)

    problems = codegen.check_required_functions(code) + codegen.check_scaffold_intact(code)
    if problems:
        return Diagnosis(
            failure_class="contract",
            route=diagnose.ROUTE_CODE,
            signature=diagnose.make_signature("contract", "scaffold", *problems),
            summary="The generated code does not satisfy the scaffold contract: "
            + "; ".join(problems),
            evidence="\n".join(problems),
            details={"problems": problems},
        )
    return None


def _preinstall(env: envman.ExperimentEnv, code: str, declared: list[str]) -> list[str]:
    """Install what the code imports, before its first run rather than after its first crash.

    The reactive path (run, fail on ImportError, install, re-run) works and is
    tested — but a first execution that dies on `import pandas` costs a whole
    cycle to learn something an AST walk knows for free.
    """
    notes: list[str] = []
    needed = codegen.extract_imports(code)
    missing = [m for m in needed if not envman.is_available(env, m)]
    if not missing:
        return notes

    packages = sorted({envman.resolve_package(m) for m in missing})
    ok, detail = envman.install(env, packages, reason="pre-flight provisioning")
    if ok:
        notes.append(f"pre-installed {packages} for imports {missing}")
    else:
        # Not fatal here: the run may not reach that import, and if it does the
        # classifier will route it to env repair with a specific module name.
        notes.append(f"pre-install of {packages} did not succeed ({detail.splitlines()[-1][:200]})")
    return notes


def run_plan(
    plan: ExperimentPlan,
    model: ChatModel,
    *,
    config: Settings | None = None,
    staging_dir: Path | None = None,
    experiments_root: Path | None = None,
) -> ExperimentReport:
    """Drive one plan from JSON to results. Never raises for an experiment failure."""
    config = config or default_settings
    started = time.time()

    root = experiments_root or Path(config.experiments_dir)
    experiment_dir = root / plan.slug
    experiment_dir.mkdir(parents=True, exist_ok=True)
    run_path = experiment_dir / "run.py"
    results_path = experiment_dir / "results.json"
    timeout = config.timeout_for(plan.complexity)

    report = ExperimentReport(
        hypothesis_id=plan.hypothesis_id,
        status=STATUS_FAILED,
        experiment_dir=str(experiment_dir),
    )

    # --- data: decide what the experiment is allowed to use, and record it ----
    requirements = data.split_requirements(
        plan.data_requirements.description, plan.data_requirements.source
    )
    sources = data.resolve(requirements, staging_dir=staging_dir, network=envman.has_network())
    report.provenance = data.write_provenance(sources, experiment_dir / "data_provenance.json")
    data_block = data.prompt_block(sources)
    logger.info(
        "[%s] data: %s", plan.hypothesis_id, report.provenance["methodological_validity"]
    )

    # --- environment ---------------------------------------------------------
    try:
        env = envman.provision(
            experiment_dir, base_env=config.base_env, venv_root=config.venv_root
        )
    except envman.EnvError as exc:
        report.status = STATUS_TERMINAL
        report.reason = f"could not provision an environment: {exc}"
        report.duration_seconds = time.time() - started
        _persist(report, experiment_dir)
        return report

    available = [m for m, ok in envman.probe(env, envman.BASE_ENV_PACKAGES).items() if ok]
    logger.info(
        "[%s] %d/%d base packages available",
        plan.hypothesis_id,
        len(available),
        len(envman.BASE_ENV_PACKAGES),
    )
    if config.base_env and len(available) < len(envman.BASE_ENV_PACKAGES) // 2:
        # Not fatal — the run proceeds and installs what it needs — but this is
        # the difference between an experiment starting in seconds and one
        # spending a quarter of an hour rebuilding the scientific stack, and it
        # is invisible unless something says so.
        logger.warning(
            "[%s] CODER_BASE_ENV=%s is configured but only %d of its packages are visible "
            "from the experiment venv. Every experiment will reinstall what the base env "
            "already has. Check `coder-agent check`.",
            plan.hypothesis_id,
            config.base_env,
            len(available),
        )

    # --- first candidate -----------------------------------------------------
    tracker = repair.ProgressTracker()
    notes: list[str] = []
    code_attempts = 0
    env_repairs = 0
    attempt_number = 0

    try:
        generated = codegen.generate(
            model,
            plan,
            data_block=data_block,
            available_packages=available,
            gpus=config.experiment_gpus,
            timeout_seconds=timeout,
            max_format_retries=config.max_format_retries,
        )
        code_attempts += 1
    except Exception as exc:
        report.status = STATUS_FAILED
        report.reason = f"the model could not produce a usable first version: {exc}"
        report.duration_seconds = time.time() - started
        _persist(report, experiment_dir)
        return report

    report.assumptions = list(generated.assumptions)
    code = generated.code

    # --- the loop ------------------------------------------------------------
    while True:
        attempt_number += 1
        attempt_started = time.time()
        run_path.write_text(code)

        diagnosis = _preflight(code)
        stage = "preflight"

        if diagnosis is None:
            notes += _preinstall(env, code, generated.requirements)
            if results_path.exists():
                results_path.unlink()  # never let a previous attempt's results look like this one's

            stage = "execute"
            logger.info("[%s] attempt %d: executing", plan.hypothesis_id, attempt_number)
            execution = executor.run_script(
                run_path,
                python_bin=env.python_bin,
                workdir=experiment_dir,
                timeout_seconds=timeout,
                memory_limit_gb=config.memory_limit_gb,
                gpus=config.experiment_gpus,
                log_dir=experiment_dir / "logs",
            )
            results, present = _read_results(results_path)
            diagnosis = diagnose.classify(
                exit_code=execution.exit_code,
                stdout=execution.stdout,
                stderr=execution.stderr,
                timed_out=execution.timed_out,
                killed_signal=execution.killed_signal,
                results=results,
                results_present=present,
                expected_metrics=list(plan.evaluation.metrics),
            )

            if diagnosis is None:
                report.status = STATUS_COMPLETED
                report.results = results
                report.attempts.append(
                    Attempt(
                        number=attempt_number,
                        stage="execute",
                        duration_seconds=time.time() - attempt_started,
                    ).to_dict()
                )
                break

        # --- something is wrong; route it ------------------------------------
        logger.info(
            "[%s] attempt %d: %s (%s)",
            plan.hypothesis_id,
            attempt_number,
            diagnosis.failure_class,
            diagnosis.route,
        )
        repair.snapshot(experiment_dir, attempt_number, code, diagnosis)

        outcome = repair.repair(
            diagnosis,
            env=env,
            code=code,
            tracker=tracker,
            data_repair_fn=None,
        )
        notes += outcome.notes

        report.attempts.append(
            Attempt(
                number=attempt_number,
                stage=stage,
                diagnosis=diagnosis.to_dict(),
                repair=outcome.to_dict(),
                duration_seconds=time.time() - attempt_started,
            ).to_dict()
        )

        if outcome.counts_against == "env":
            env_repairs += 1

        if outcome.terminal:
            report.status = (
                STATUS_NO_PROGRESS if outcome.stop_kind == "no_progress" else STATUS_TERMINAL
            )
            report.reason = outcome.detail
            break

        if env_repairs > config.max_env_repairs:
            report.status = STATUS_BUDGET_EXHAUSTED
            report.reason = (
                f"environment repairs exhausted after {env_repairs} installs; "
                "the base environment is missing too much for this experiment"
            )
            break

        if outcome.code is not None:
            code = outcome.code          # downscale rewrote it deterministically
            continue

        if not outcome.regenerate:
            continue                     # install / data fix: re-run unchanged code

        if code_attempts >= config.max_code_attempts:
            report.status = STATUS_BUDGET_EXHAUSTED
            report.reason = (
                f"code attempts exhausted after {code_attempts} versions; "
                f"last failure: {diagnosis.summary}"
            )
            break

        try:
            generated = codegen.regenerate(
                model,
                plan,
                code=code,
                diagnosis_summary=diagnosis.summary,
                failure_class=diagnosis.failure_class,
                evidence=diagnosis.evidence,
                data_block=data_block,
                available_packages=available,
                gpus=config.experiment_gpus,
                timeout_seconds=timeout,
                attempt=attempt_number,
                escalate=outcome.escalate,
                previous_notes=notes[-6:],
                max_format_retries=config.max_format_retries,
            )
            code_attempts += 1
            code = generated.code
            report.assumptions = list(generated.assumptions)
        except Exception as exc:
            report.status = STATUS_FAILED
            report.reason = f"regeneration failed: {exc}"
            break

    # --- finish --------------------------------------------------------------
    report.code_attempts_used = code_attempts
    report.env_repairs_used = env_repairs
    report.duration_seconds = time.time() - started
    report.env = env.to_dict()
    envman.write_manifest(env, experiment_dir / "environment.json")

    report.evaluation = data.stamp_evaluation(
        report.results or {},
        sources,
        success_criteria=plan.evaluation.success_criteria,
        refute_criteria=plan.evaluation.refute_criteria,
    )
    if report.status == STATUS_COMPLETED and not report.reason:
        report.reason = "ran to completion and produced the planned metrics"

    _persist(report, experiment_dir)
    logger.info(
        "[%s] %s — %s", plan.hypothesis_id, report.status, report.evaluation.get("hypothesis_outcome")
    )
    return report


def _persist(report: ExperimentReport, experiment_dir: Path) -> None:
    (experiment_dir / "run_report.json").write_text(json.dumps(report.to_dict(), indent=2))


def run_all(
    plans: list[ExperimentPlan],
    model: ChatModel,
    *,
    config: Settings | None = None,
    staging_dir: Path | None = None,
    experiments_root: Path | None = None,
) -> dict[str, Any]:
    """Run every plan in order, and write the run-level summary.

    One plan failing never stops the others: a summary covering four experiments,
    three of which worked, is worth far more than an exception at the second.
    """
    config = config or default_settings
    root = experiments_root or Path(config.experiments_dir)
    root.mkdir(parents=True, exist_ok=True)

    reports = []
    for plan in plans:
        try:
            report = run_plan(
                plan, model, config=config, staging_dir=staging_dir, experiments_root=root
            )
        except Exception as exc:  # a harness bug must not lose the other plans
            logger.exception("harness error while running %s", plan.hypothesis_id)
            report = ExperimentReport(
                hypothesis_id=plan.hypothesis_id,
                status=STATUS_FAILED,
                experiment_dir=str(root / plan.slug),
                reason=f"harness error: {type(exc).__name__}: {exc}",
            )
        reports.append(report)

    summary = {
        "experiments": [r.to_dict() for r in reports],
        "completed": sum(1 for r in reports if r.status == STATUS_COMPLETED),
        "total": len(reports),
        "model": config.llm_model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
