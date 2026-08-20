"""Turn a plan into a runnable `run.py`, and check it before anything executes.

The model never writes the whole file. It fills named sections that are spliced
into `templates/run.py.template`, so the metadata header and the orchestration
footer — the parts that guarantee a results.json exists — are ours and cannot be
rewritten by a bad generation.

Splicing is `str.replace()` on `__TOKEN__` markers, not `str.format()`: generated
Python routinely contains literal `{` and `}` (every dict, every f-string), and
format-string substitution on it fails on a correct response.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import prompts
from .llm import ChatModel
from .plan import ExperimentPlan
from .sections import SectionParseError, parse_sections

TEMPLATE_PATH = Path(__file__).parent / "templates" / "run.py.template"


@dataclass
class GeneratedCode:
    """One complete candidate: the file to run, plus what the model said about it."""

    code: str
    sections: dict[str, str]
    requirements: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)


def load_template() -> str:
    return TEMPLATE_PATH.read_text()


def render(plan: ExperimentPlan, sections: dict[str, str]) -> str:
    """Splice the model's sections into the fixed scaffold."""
    code = load_template()
    code = code.replace("__EXPERIMENT_ID__", plan.hypothesis_id)
    code = code.replace("__OBJECTIVE__", plan.objective.replace('"""', "'''"))
    for field_name in prompts.CODE_FIELDS:
        token = "__" + field_name.upper() + "__"
        code = code.replace(token, sections.get(field_name, "").strip() or f"# (no {field_name})")
    return code


def parse_requirements(text: str) -> list[str]:
    """Turn the `requirements` section into a clean list."""
    if not text:
        return []
    out = []
    for line in text.splitlines():
        line = line.strip().lstrip("-* ").strip()
        if not line or line.startswith("#") or line.lower() in {"none", "n/a", "(none)"}:
            continue
        out.append(line)
    return out


def parse_assumptions(text: str) -> list[str]:
    if not text:
        return []
    return [ln.strip().lstrip("-* ").strip() for ln in text.splitlines() if ln.strip()]


def extract_imports(code: str) -> list[str]:
    """Top-level module names the file imports, via AST.

    Used to pre-provision the environment *before* the first execution, so the
    common missing-package case is handled without a failed run at all. AST
    rather than regex: a module name inside a docstring or a comment is not an
    import, and installing it would be noise at best.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return sorted(m for m in modules if m not in sys.stdlib_module_names)


def check_required_functions(code: str) -> list[str]:
    """Which contract functions are missing or have the wrong arity.

    Caught here rather than at runtime because the failure mode otherwise is a
    NameError from the scaffold, which reads like a scaffold bug and sends the
    fix loop looking in the wrong place.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # the compile check reports this properly

    defined = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    expected_arity = {
        "load_data": 0, "preprocess": 1, "build_model": 1,
        "run_experiment": 2, "evaluate": 2,
    }

    problems = []
    for name, signature in prompts.REQUIRED_FUNCTIONS.items():
        node = defined.get(name)
        if node is None:
            problems.append(f"{name} is not defined at module level (the scaffold calls {signature})")
            continue
        required = len(node.args.args) - len(node.args.defaults)
        wanted = expected_arity[name]
        if required > wanted or (len(node.args.args) < wanted and not node.args.vararg):
            problems.append(
                f"{name} takes {len(node.args.args)} argument(s); the scaffold calls {signature}"
            )
    return problems


def check_scaffold_intact(code: str) -> list[str]:
    """The generated sections must not redefine what the footer owns."""
    problems = []
    if code.count('if __name__ == "__main__"') > 1 or code.count("if __name__ == '__main__'") > 0:
        problems.append("the response defines its own __main__ block; the scaffold already has one")
    if code.count("def main(") > 1:
        problems.append("the response defines its own main(); the scaffold already has one")
    if "RESULTS_PATH.write_text" in code and code.count("RESULTS_PATH.write_text") > 1:
        problems.append("the response writes results.json itself; only the scaffold may do that")
    return problems


def build(plan: ExperimentPlan, sections: dict[str, str]) -> GeneratedCode:
    """Assemble a `GeneratedCode` from parsed sections."""
    code = render(plan, sections)
    return GeneratedCode(
        code=code,
        sections=sections,
        requirements=parse_requirements(sections.get("requirements", "")),
        assumptions=parse_assumptions(sections.get("assumptions", "")),
        imports=extract_imports(code),
    )


def call_model(
    model: ChatModel,
    prompt: str,
    *,
    field_names: list[str],
    max_format_retries: int = 2,
) -> dict[str, str]:
    """Send a prompt and get parsed sections back, with one targeted format repair.

    A malformed response is a transport failure, not a code failure, so it gets
    its own small retry budget instead of consuming a code attempt — the previous
    system spent a third of its fix budget on exactly this.
    """
    response = model.complete(prompt, system=prompts.SYSTEM)
    last_error: Exception | None = None

    for _ in range(max(0, max_format_retries)):
        try:
            return parse_sections(response, required=field_names)
        except SectionParseError as exc:
            last_error = exc
            response = model.complete(
                prompts.repair_format_prompt(str(exc), field_names), system=prompts.SYSTEM
            )

    return parse_sections(response, required=field_names)  # raises with the final error


def generate(
    model: ChatModel,
    plan: ExperimentPlan,
    *,
    data_block: str,
    available_packages: list[str],
    gpus: str,
    timeout_seconds: int,
    max_format_retries: int = 2,
) -> GeneratedCode:
    prompt = prompts.codegen_prompt(
        plan,
        data_block=data_block,
        available_packages=available_packages,
        gpus=gpus,
        timeout_seconds=timeout_seconds,
    )
    sections = call_model(
        model, prompt, field_names=prompts.EXPERIMENT_FIELD_NAMES, max_format_retries=max_format_retries
    )
    return build(plan, sections)


def regenerate(
    model: ChatModel,
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
    escalate: bool = False,
    previous_notes: list[str] | None = None,
    max_format_retries: int = 2,
) -> GeneratedCode:
    prompt = prompts.fix_prompt(
        plan,
        code=code,
        diagnosis_summary=diagnosis_summary,
        failure_class=failure_class,
        evidence=evidence,
        data_block=data_block,
        available_packages=available_packages,
        gpus=gpus,
        timeout_seconds=timeout_seconds,
        attempt=attempt,
        escalate=escalate,
        previous_notes=previous_notes,
    )
    sections = call_model(
        model, prompt, field_names=prompts.EXPERIMENT_FIELD_NAMES, max_format_retries=max_format_retries
    )
    return build(plan, sections)


def normalized(code: str) -> str:
    """Source stripped to what actually executes, for comparing two candidates.

    Comments, blank lines and indentation depth are removed, so a regeneration
    that only reworded a comment or hoisted an expression into a variable
    compares equal to what it replaced.
    """
    import io
    import tokenize

    try:
        pieces = []
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE):
                continue
            if token.type == tokenize.INDENT or token.type == tokenize.DEDENT:
                continue
            pieces.append(token.string.strip())
        return " ".join(p for p in pieces if p)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Unparseable source is compared verbatim; the compile check will
        # reject it moments later anyway.
        return " ".join(code.split())


def is_cosmetic_change(before: str, after: str) -> bool:
    """Did a regeneration change anything that will alter execution?

    A model asked to fix a failure it does not understand will often return the
    same logic with a fresh comment explaining the problem it did not solve.
    Executing that costs a full run to rediscover the identical error, so it is
    worth detecting before the subprocess starts rather than after.
    """
    return normalized(before) == normalized(after)
