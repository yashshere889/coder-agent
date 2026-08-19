"""Route a diagnosis to the thing that can actually fix it.

Four repairers, and only one of them is the code generator:

    env_repair        install the package, re-run the *unchanged* code
    data_repair       re-resolve the input, or fall down to a declared surrogate
    downscale         shrink the run deterministically (no model call at all)
    code_regeneration ask the model for a new version, with the failure quoted

Two of these cost nothing but a subprocess, which is the point: the cheap,
certain repairs get tried before the expensive, uncertain one, and only genuine
code defects consume the code-attempt budget.

`ProgressTracker` is the guard against the other observed failure mode — the
loop hitting the same wall three times. It watches diagnosis signatures and
escalates strategy on a repeat rather than letting the budget drain into an
identical retry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import diagnose, envman
from .diagnose import Diagnosis
from .envman import ExperimentEnv

# Knobs that make a run smaller without changing what it measures. Halving
# `draws` costs posterior precision; it does not turn the experiment into a
# different experiment — which is why this is a legitimate automatic repair and
# changing, say, the model formula would not be.
DOWNSCALE_KNOBS: dict[str, int] = {
    "draws": 250,
    "tune": 250,
    "chains": 2,
    "iter_sampling": 250,
    "iter_warmup": 250,
    "n_samples": 500,
    "num_samples": 500,
    "n_iter": 100,
    "max_iter": 100,
    "epochs": 1,
    "batch_size": 8,
    "n_estimators": 50,
    "n_rows": 1000,
    "sample_size": 1000,
    "max_rows": 1000,
    "n_boot": 100,
    "bootstrap_samples": 100,
}

# Repairs that must not silently continue: whatever the loop does next, a human
# has to see these in the report.
TERMINAL_CLASSES = {"missing_system_library"}


@dataclass
class RepairOutcome:
    """What the router did, and what the loop should do next."""

    action: str
    ok: bool
    detail: str
    code: str | None = None          # set when the repairer rewrote the source itself
    regenerate: bool = False         # ask the model for new code
    terminal: bool = False           # stop; nothing here can fix it
    stop_kind: str = ""              # "no_progress" | "unfixable", when terminal
    escalate: bool = False           # regenerate, but demand a different approach
    counts_against: str = ""         # "code" | "env" | "" (free)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "ok": self.ok,
            "detail": self.detail[:2000],
            "regenerate": self.regenerate,
            "terminal": self.terminal,
            "stop_kind": self.stop_kind,
            "escalate": self.escalate,
            "counts_against": self.counts_against,
            "notes": self.notes,
        }


class ProgressTracker:
    """Detects the loop re-running into the same wall.

    First repeat escalates: the next code request must take a *different*
    approach rather than patch the same one. Second repeat stops the experiment
    — three identical failures is the exact shape of the run this agent was
    built to not reproduce, and continuing past it only spends allocation to
    reach the same place.
    """

    def __init__(self, *, escalate_after: int = 2, stop_after: int = 3) -> None:
        self.escalate_after = escalate_after
        self.stop_after = stop_after
        self.history: list[str] = []

    def record(self, signature: str) -> None:
        self.history.append(signature)

    def count(self, signature: str) -> int:
        return self.history.count(signature)

    def should_escalate(self, signature: str) -> bool:
        return self.count(signature) >= self.escalate_after

    def should_stop(self, signature: str) -> bool:
        return self.count(signature) >= self.stop_after

    def repeated_consecutively(self) -> bool:
        return len(self.history) >= 2 and self.history[-1] == self.history[-2]


def repair(
    diagnosis: Diagnosis,
    *,
    env: ExperimentEnv,
    code: str,
    tracker: ProgressTracker,
    data_repair_fn=None,
) -> RepairOutcome:
    """Dispatch one diagnosis. Never raises — a failed repair is an outcome, not an exception."""
    tracker.record(diagnosis.signature)

    if diagnosis.failure_class in TERMINAL_CLASSES:
        return RepairOutcome(
            action="stop",
            ok=False,
            detail=diagnosis.summary,
            terminal=True,
            stop_kind="unfixable",
            notes=["needs a cluster module or a rebuilt base env; not fixable from inside the run"],
        )

    if tracker.should_stop(diagnosis.signature):
        return RepairOutcome(
            action="stop",
            ok=False,
            detail=(
                f"The same failure has now occurred {tracker.count(diagnosis.signature)} times "
                f"({diagnosis.summary}). Stopping instead of spending the remaining budget on it."
            ),
            terminal=True,
            stop_kind="no_progress",
            notes=["no-progress guard tripped"],
        )

    if diagnosis.route == diagnose.ROUTE_ENV:
        return _repair_env(diagnosis, env)
    if diagnosis.route == diagnose.ROUTE_DATA:
        return _repair_data(diagnosis, data_repair_fn)
    if diagnosis.route == diagnose.ROUTE_DOWNSCALE:
        return _repair_downscale(diagnosis, code, tracker)

    # Everything else is a genuine code defect.
    return RepairOutcome(
        action="regenerate",
        ok=True,
        detail=diagnosis.summary,
        regenerate=True,
        escalate=tracker.should_escalate(diagnosis.signature),
        counts_against="code",
    )


def _repair_env(diagnosis: Diagnosis, env: ExperimentEnv) -> RepairOutcome:
    """Install the missing package and re-run what we already have.

    Deliberately does NOT set `regenerate`: the source was never the problem, and
    handing it to the model would only introduce new ways for it to be wrong.
    """
    module = diagnosis.module or ""
    package = envman.resolve_package(module)
    ok, detail = envman.install(env, [package], reason=f"missing import {module!r}")

    if not ok and package != module:
        # The mapping guessed wrong; the import name itself is the next best bet.
        ok, detail = envman.install(env, [module], reason=f"fallback for {module!r}")
        package = module

    if not ok:
        return RepairOutcome(
            action="install",
            ok=False,
            detail=f"could not install {package!r} for import {module!r}: {detail[-1200:]}",
            terminal=True,
            stop_kind="unfixable",
            counts_against="env",
            notes=[f"add {package!r} to scripts/build_base_env.sh if this recurs"],
        )

    return RepairOutcome(
        action="install",
        ok=True,
        detail=f"installed {package!r} to satisfy `import {module}`; re-running unchanged code",
        counts_against="env",
        notes=["code untouched — this was an environment failure, not a code failure"],
    )


def _repair_data(diagnosis: Diagnosis, data_repair_fn) -> RepairOutcome:
    """Hand back to the data layer to re-resolve the input or declare a surrogate."""
    if data_repair_fn is None:
        return RepairOutcome(
            action="regenerate",
            ok=True,
            detail=f"{diagnosis.summary} (no data resolver wired; asking for guarded loading instead)",
            regenerate=True,
            counts_against="code",
        )
    try:
        ok, detail = data_repair_fn(diagnosis)
    except Exception as exc:  # a resolver failure must not kill the run
        ok, detail = False, f"data resolver raised {type(exc).__name__}: {exc}"

    if ok:
        return RepairOutcome(
            action="resolve_data", ok=True, detail=detail,
            notes=["input re-resolved; re-running unchanged code"],
        )
    return RepairOutcome(
        action="resolve_data",
        ok=False,
        detail=detail,
        regenerate=True,
        counts_against="code",
        notes=["data could not be resolved — regenerating with an explicit surrogate path"],
    )


def _repair_downscale(diagnosis: Diagnosis, code: str, tracker: ProgressTracker) -> RepairOutcome:
    """Shrink the run deterministically before asking the model to rethink it.

    An OOM or a timeout usually means the code is right and the numbers are too
    big. Halving them is a transformation we can make correctly in Python, for
    free, without a model round-trip and without the risk of getting a working
    experiment rewritten into a broken one.
    """
    shrunk, changes = downscale(code)
    if not changes:
        return RepairOutcome(
            action="regenerate",
            ok=True,
            detail=f"{diagnosis.summary} No downscalable parameters found in the code.",
            regenerate=True,
            escalate=True,
            counts_against="code",
            notes=["ask for an approach that fits the budget, not the same one shrunk"],
        )
    return RepairOutcome(
        action="downscale",
        ok=True,
        detail=f"{diagnosis.summary} Reduced: " + "; ".join(changes),
        code=shrunk,
        notes=changes + ["scale reduced deterministically — no model call"],
    )


def downscale(code: str) -> tuple[str, list[str]]:
    """Halve the known cost knobs. Returns (new_code, human-readable changes).

    Only touches assignments and keyword arguments whose *name* is a known knob,
    so a literal `2000` that happens to be a year is left alone. Each knob has a
    floor: shrinking `chains=4` to `chains=0` would not run at all.
    """
    changes: list[str] = []
    result = code

    for knob, floor in DOWNSCALE_KNOBS.items():
        pattern = re.compile(
            rf"(?P<prefix>\b{re.escape(knob)}\b\s*(?P<op>=)\s*)(?P<value>\d+)",
            re.IGNORECASE,
        )

        def shrink(match: re.Match) -> str:
            current = int(match.group("value"))
            reduced = max(floor, current // 2)
            if reduced >= current:
                return match.group(0)
            changes.append(f"{knob}: {current} -> {reduced}")
            return f"{match.group('prefix')}{reduced}"

        result = pattern.sub(shrink, result)

    return result, changes


def snapshot(experiment_dir: Path, attempt: int, code: str, diagnosis: Diagnosis | None) -> Path:
    """Keep every failed attempt on disk, next to why it failed.

    A run that ends unsuccessfully after four attempts is still evidence; the
    summary alone never explains what the model kept getting wrong.
    """
    target = experiment_dir / "attempts" / f"attempt_{attempt}"
    target.mkdir(parents=True, exist_ok=True)
    (target / "run.py").write_text(code)
    if diagnosis:
        import json

        (target / "diagnosis.json").write_text(json.dumps(diagnosis.to_dict(), indent=2))
        (target / "evidence.txt").write_text(diagnosis.evidence)
    return target
