"""Parse and validate the experiment plan JSON.

The agent's whole input contract is one file on disk, so this module is the
only place that knows its shape. Everything downstream takes an
`ExperimentPlan`, never a raw dict — a plan whose `objective` is missing should
fail here, loudly, before a GPU is allocated, rather than as a KeyError inside a
prompt template forty minutes later.

Unknown keys are preserved in `raw` rather than rejected: the plan is produced
by an upstream tool that will grow fields, and refusing to run because of a new
one would be a bad trade.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_COMPLEXITY = {"low", "medium", "high"}


class PlanError(ValueError):
    """The plan file is missing something the agent cannot proceed without."""


@dataclass(frozen=True)
class DataRequirement:
    """One data input the experiment needs, before any attempt to obtain it.

    `source` and `description` are free text from the planner; `preprocessing_steps`
    is the ordered list it wants applied. Resolution into an actual file (real or
    surrogate) is `data.py`'s job, not this module's.
    """

    source: str
    description: str
    preprocessing_steps: tuple[str, ...] = ()


@dataclass(frozen=True)
class Evaluation:
    metrics: tuple[str, ...]
    baseline: str
    success_criteria: str
    refute_criteria: str


@dataclass(frozen=True)
class ExperimentPlan:
    hypothesis_id: str
    objective: str
    design: str
    feasible: bool
    complexity: str
    independent_variables: tuple[str, ...]
    dependent_variables: tuple[str, ...]
    methods: tuple[dict[str, Any], ...]
    data_requirements: DataRequirement
    evaluation: Evaluation
    implementation_steps: tuple[str, ...]
    risks: tuple[str, ...]
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def slug(self) -> str:
        """Directory-safe id. The planner controls `hypothesis_id`, so don't trust it as a path."""
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.hypothesis_id)
        return safe or "experiment"


def _require_str(obj: dict[str, Any], key: str, where: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"{where}: '{key}' must be a non-empty string, got {value!r}")
    return value.strip()


def _str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(v).strip() for v in value if str(v).strip())
    raise PlanError(f"expected a list of strings, got {type(value).__name__}")


def _parse_implementation_steps(value: Any) -> tuple[str, ...]:
    """Accept both `[{"step": 1, "description": "..."}]` and a plain list of strings.

    The example plans use the former; hand-written ones tend to use the latter,
    and there is no reason to make a human write the wrapper.
    """
    if not value:
        return ()
    steps: list[tuple[int, str]] = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, dict):
            description = str(item.get("description", "")).strip()
            order = item.get("step", index)
            try:
                order = int(order)
            except (TypeError, ValueError):
                order = index
        else:
            description, order = str(item).strip(), index
        if description:
            steps.append((order, description))
    steps.sort(key=lambda pair: pair[0])
    return tuple(description for _, description in steps)


def parse_plan(obj: dict[str, Any]) -> ExperimentPlan:
    """Turn one entry of `experiment_plans` into an `ExperimentPlan`."""
    if not isinstance(obj, dict):
        raise PlanError(f"an experiment plan must be an object, got {type(obj).__name__}")

    hypothesis_id = _require_str(obj, "hypothesis_id", "plan")
    where = f"plan {hypothesis_id}"

    complexity = str(obj.get("estimated_complexity", "medium")).strip().lower()
    if complexity not in VALID_COMPLEXITY:
        # Not fatal: an unrecognized complexity just means the default timeout.
        complexity = "medium"

    variables = obj.get("variables") or {}
    if not isinstance(variables, dict):
        raise PlanError(f"{where}: 'variables' must be an object")

    raw_data = obj.get("data_requirements") or {}
    if not isinstance(raw_data, dict):
        raise PlanError(f"{where}: 'data_requirements' must be an object")

    raw_eval = obj.get("evaluation") or {}
    if not isinstance(raw_eval, dict):
        raise PlanError(f"{where}: 'evaluation' must be an object")
    metrics = _str_tuple(raw_eval.get("metrics"))
    if not metrics:
        raise PlanError(
            f"{where}: 'evaluation.metrics' is empty — there is nothing for the "
            "experiment to compute, so there is nothing to check it against"
        )

    methods = tuple(m for m in (obj.get("methods") or []) if isinstance(m, dict))

    return ExperimentPlan(
        hypothesis_id=hypothesis_id,
        objective=_require_str(obj, "objective", where),
        design=str(obj.get("design", "")).strip(),
        feasible=bool(obj.get("feasible", True)),
        complexity=complexity,
        independent_variables=_str_tuple(variables.get("independent")),
        dependent_variables=_str_tuple(variables.get("dependent")),
        methods=methods,
        data_requirements=DataRequirement(
            source=str(raw_data.get("source", "")).strip(),
            description=str(raw_data.get("description", "")).strip(),
            preprocessing_steps=_str_tuple(raw_data.get("preprocessing_steps")),
        ),
        evaluation=Evaluation(
            metrics=metrics,
            baseline=str(raw_eval.get("baseline", "")).strip(),
            success_criteria=str(raw_eval.get("success_criteria", "")).strip(),
            refute_criteria=str(raw_eval.get("refute_criteria", "")).strip(),
        ),
        implementation_steps=_parse_implementation_steps(obj.get("implementation_steps")),
        risks=_str_tuple(obj.get("risks")),
        raw=obj,
    )


def load_plans(path: str | Path, hypothesis_ids: list[str] | None = None) -> list[ExperimentPlan]:
    """Read a plan file and return its plans, in the planner's own priority order.

    Infeasible plans are dropped: the planner already decided they cannot be run
    within the compute and data constraints, and generating code for one anyway
    would spend a GPU allocation to reach a conclusion the input already states.
    """
    path = Path(path)
    try:
        document = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise PlanError(f"plan file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PlanError(f"plan file is not valid JSON ({path}): {exc}") from exc

    entries = document.get("experiment_plans")
    if not isinstance(entries, list) or not entries:
        raise PlanError(f"{path}: 'experiment_plans' must be a non-empty list")

    plans = [parse_plan(entry) for entry in entries]

    ranks = {}
    for item in document.get("priority_order") or []:
        if isinstance(item, dict) and "hypothesis_id" in item:
            try:
                ranks[str(item["hypothesis_id"])] = int(item.get("rank", 10**6))
            except (TypeError, ValueError):
                continue
    plans.sort(key=lambda p: ranks.get(p.hypothesis_id, 10**6))

    if hypothesis_ids:
        wanted = set(hypothesis_ids)
        missing = wanted - {p.hypothesis_id for p in plans}
        if missing:
            raise PlanError(f"{path}: no plan for hypothesis id(s) {sorted(missing)}")
        plans = [p for p in plans if p.hypothesis_id in wanted]

    runnable = [p for p in plans if p.feasible]
    if not runnable:
        raise PlanError(f"{path}: every selected plan is marked feasible=false")
    return runnable
