"""Every prompt the agent sends, built as functions rather than format strings.

Generated Python is full of literal braces, and so is the evidence quoted back
into a fix prompt. `str.format()` on a template containing either raises at call
time, not at import — a failure that shows up mid-run, on the cluster, an hour
in. So prompts are assembled by concatenation and the templating problem simply
does not exist.

The field list below is the single source of truth for the response shape: the
format example shown to the model is *generated* from it, so the shape asked for
cannot drift from the shape `sections.parse_sections` accepts.
"""

from __future__ import annotations

import json

from .plan import ExperimentPlan
from .sections import format_template

# The sections a codegen response must carry. Order is the order they are
# spliced into the template.
CODE_FIELDS = [
    "imports",
    "configuration",
    "helpers",
    "load_data",
    "preprocess",
    "build_model",
    "run_experiment",
    "evaluate",
]
META_FIELDS = ["requirements", "assumptions"]
EXPERIMENT_FIELD_NAMES = CODE_FIELDS + META_FIELDS

# Signatures the orchestration footer calls. A response that renames one of
# these produces a NameError at runtime, so it is checked before execution.
REQUIRED_FUNCTIONS = {
    "load_data": "load_data()",
    "preprocess": "preprocess(raw)",
    "build_model": "build_model(prepared)",
    "run_experiment": "run_experiment(model, prepared)",
    "evaluate": "evaluate(outputs, prepared)",
}

SYSTEM = (
    "You are a careful research engineer. You write complete, runnable Python for "
    "scientific experiments on an HPC cluster. You never write placeholder code, never "
    "leave a TODO, and never invent data or results. When you are unsure about an input, "
    "you say so in the assumptions section rather than guessing silently in the code."
)


def _contract_block() -> str:
    lines = [
        "CONTRACT — the scaffold calls exactly these functions, in this order:",
        "",
        "    raw       = load_data()",
        "    prepared  = preprocess(raw)",
        "    model     = build_model(prepared)",
        "    outputs   = run_experiment(model, prepared)",
        "    evaluation= evaluate(outputs, prepared)",
        "",
        "Define every one of them with exactly those names and arities. `evaluate` must "
        "return a dict shaped:",
        "",
        '    {"metrics": {...}, "hypothesis_outcome": "supported"|"refuted"|"inconclusive",',
        '     "meets_success_criteria": bool, "notes": str}',
        "",
        "`metrics` must contain a real number (or list of numbers) for each metric the plan "
        "names. Never return an empty metrics dict, and never return zeros or NaN as a "
        "stand-in for a computation you did not do — the harness checks for exactly that and "
        "will reject the run.",
        "",
        "Do NOT write imports, logging setup, a main(), a __main__ block, or anything that "
        "writes results.json. The scaffold already does all of that; duplicating it breaks it.",
    ]
    return "\n".join(lines)


def _plan_block(plan: ExperimentPlan) -> str:
    methods = "\n".join(
        f"  - {m.get('name', 'method')}: {m.get('description', '')}" for m in plan.methods
    ) or "  (none specified)"
    steps = "\n".join(f"  {i}. {s}" for i, s in enumerate(plan.implementation_steps, 1)) or "  (none)"
    preprocessing = "\n".join(f"  - {s}" for s in plan.data_requirements.preprocessing_steps) or "  (none)"
    metrics = "\n".join(f"  - {m}" for m in plan.evaluation.metrics)

    return "\n".join(
        [
            f"EXPERIMENT PLAN ({plan.hypothesis_id})",
            "",
            f"Objective: {plan.objective}",
            f"Design: {plan.design}",
            f"Complexity: {plan.complexity}",
            "",
            f"Independent variables: {', '.join(plan.independent_variables) or '(none)'}",
            f"Dependent variables: {', '.join(plan.dependent_variables) or '(none)'}",
            "",
            "Methods:",
            methods,
            "",
            "Preprocessing the plan asks for:",
            preprocessing,
            "",
            "Implementation steps:",
            steps,
            "",
            "Metrics that MUST appear in evaluate()'s output:",
            metrics,
            "",
            f"Baseline: {plan.evaluation.baseline or '(none stated)'}",
            f"Success criteria: {plan.evaluation.success_criteria or '(none stated)'}",
            f"Refute criteria: {plan.evaluation.refute_criteria or '(none stated)'}",
        ]
    )


def _environment_block(available: list[str], gpus: str, timeout_seconds: int) -> str:
    return "\n".join(
        [
            "ENVIRONMENT",
            "",
            f"Already installed (import these freely): {', '.join(available) or '(nothing verified)'}",
            "Anything else you import will be installed automatically before the run is retried — "
            "so import what you genuinely need, but prefer the list above, and list every "
            "non-stdlib import in the `requirements` section.",
            "",
            f"GPUs visible to this experiment: {gpus or 'none — write CPU-only code'}",
            f"Wall-clock budget: {timeout_seconds} seconds. Size the run to finish well inside it; "
            "a run killed at the limit produces nothing at all.",
            "",
            "The compute node may have no outbound network. Do not assume you can download "
            "anything that is not listed in the DATA section below.",
        ]
    )


def _format_block() -> str:
    return "\n".join(
        [
            "RESPONSE FORMAT — reply with exactly these sections and nothing outside them:",
            "",
            format_template(EXPERIMENT_FIELD_NAMES),
            "",
            "Rules for the format:",
            "  - Plain Python inside the code sections. No markdown fences, no commentary.",
            "  - `imports` holds import statements only.",
            "  - `configuration` holds module-level constants (seeds, paths, sizes).",
            "  - `helpers` holds any extra functions the five contract functions call. It may be "
            "empty, but the section must still be present.",
            "  - `requirements` is one pip requirement per line (or the single word `none`).",
            "  - `assumptions` is one plain-English assumption per line — anything you had to "
            "decide that the plan did not specify.",
        ]
    )


def codegen_prompt(
    plan: ExperimentPlan,
    *,
    data_block: str,
    available_packages: list[str],
    gpus: str,
    timeout_seconds: int,
) -> str:
    return "\n\n".join(
        [
            "Write the experiment described by this plan.",
            _plan_block(plan),
            "DATA\n\n" + data_block,
            _environment_block(available_packages, gpus, timeout_seconds),
            _contract_block(),
            _format_block(),
        ]
    )


def fix_prompt(
    plan: ExperimentPlan,
    *,
    code: str,
    diagnosis_summary: str,
    failure_class: str,
    evidence: str,
    data_block: str,
    available_packages: list[str],
    gpus: str,
    timeout_seconds: int,
    attempt: int,
    escalate: bool,
    previous_notes: list[str] | None = None,
) -> str:
    """Ask for a corrected version, quoting the concrete failure.

    `escalate` is set when the same failure has already recurred. It changes the
    request from "fix this" to "solve it a different way", because a second patch
    to an approach that has now failed twice is overwhelmingly a third failure.
    """
    header = [
        f"The experiment you wrote failed on attempt {attempt}. Here is what happened.",
        "",
        f"Failure class: {failure_class}",
        f"Diagnosis: {diagnosis_summary}",
    ]

    if escalate:
        header += [
            "",
            "IMPORTANT: this is the SECOND time this exact failure has occurred. Do not patch "
            "the same approach again — it has now been tried and it does not work. Choose a "
            "different implementation strategy for the part that keeps failing: a simpler "
            "method, a different library, a smaller formulation of the same question. State "
            "what you changed and why in the `assumptions` section.",
        ]

    if previous_notes:
        header += ["", "Repairs already applied by the harness:"] + [f"  - {n}" for n in previous_notes]

    return "\n\n".join(
        [
            "\n".join(header),
            "EVIDENCE (from the failed run)\n\n" + evidence.strip()[-6000:],
            "THE CODE THAT FAILED\n\n" + code.strip()[-24000:],
            _plan_block(plan),
            "DATA\n\n" + data_block,
            _environment_block(available_packages, gpus, timeout_seconds),
            _contract_block(),
            "Return the COMPLETE corrected version of every section — not a diff, not only the "
            "part you changed. The sections are spliced into a fixed scaffold, so a partial "
            "response produces a broken file.",
            _format_block(),
        ]
    )


def repair_format_prompt(error: str, field_names: list[str]) -> str:
    """One targeted retry when the response did not parse. Cheaper than a full regeneration."""
    return "\n\n".join(
        [
            "Your previous response could not be parsed:",
            str(error),
            "Resend the SAME content in exactly this format. Every section must be present, "
            "the delimiters must match exactly, and nothing may appear outside them:",
            format_template(field_names),
        ]
    )


def plan_summary_for_log(plan: ExperimentPlan) -> str:
    return json.dumps(
        {
            "hypothesis_id": plan.hypothesis_id,
            "objective": plan.objective[:200],
            "complexity": plan.complexity,
            "metrics": list(plan.evaluation.metrics),
        },
        indent=2,
    )
