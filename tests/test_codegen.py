"""Splicing model sections into the fixed scaffold, and checking the result."""

from __future__ import annotations

from coder_agent import codegen, prompts
from coder_agent.sections import SectionParseError, render_sections

from .conftest import FakeModel, working_sections


def test_generated_code_compiles_and_keeps_the_scaffold(plan):
    built = codegen.build(plan, working_sections())
    assert codegen.check_required_functions(built.code) == []
    assert codegen.check_scaffold_intact(built.code) == []
    compile(built.code, "run.py", "exec")


def test_literal_braces_in_generated_code_survive_splicing(plan):
    """The reason splicing is str.replace and not str.format."""
    sections = working_sections(body="return {'x': 1.5, 'meta': {'nested': f'{1+1}'}}")
    built = codegen.build(plan, sections)
    assert "{'nested': f'{1+1}'}" in built.code
    compile(built.code, "run.py", "exec")


def test_a_missing_contract_function_is_caught_before_execution(plan):
    sections = working_sections()
    sections["evaluate"] = "def scoring(outputs, prepared):\n    return {}"
    built = codegen.build(plan, sections)

    problems = codegen.check_required_functions(built.code)
    assert any("evaluate is not defined" in p for p in problems)


def test_a_contract_function_with_the_wrong_arity_is_caught(plan):
    sections = working_sections()
    sections["run_experiment"] = "def run_experiment(model, prepared, extra):\n    return {}"
    built = codegen.build(plan, sections)
    assert any("run_experiment takes" in p for p in codegen.check_required_functions(built.code))


def test_a_response_that_rewrites_the_orchestration_is_rejected(plan):
    sections = working_sections()
    sections["helpers"] = "def main():\n    pass\n\nif __name__ == '__main__':\n    main()"
    built = codegen.build(plan, sections)
    assert codegen.check_scaffold_intact(built.code)


def test_imports_are_extracted_from_the_ast_not_from_prose(plan):
    sections = working_sections(
        imports="import pandas as pd\nimport numpy as np\nfrom sklearn.linear_model import Ridge"
    )
    sections["configuration"] = "# we deliberately do not import tensorflow here\nSEED = 1"
    built = codegen.build(plan, sections)

    assert "pandas" in built.imports and "numpy" in built.imports and "sklearn" in built.imports
    assert "tensorflow" not in built.imports
    assert "json" not in built.imports, "stdlib modules never need installing"


def test_requirements_and_assumptions_are_parsed(plan):
    sections = working_sections()
    sections["requirements"] = "pandas>=2.0\n- numpy\n# a comment\n"
    sections["assumptions"] = "- Population weights are uniform\nInteractions are products"
    built = codegen.build(plan, sections)

    assert built.requirements == ["pandas>=2.0", "numpy"]
    assert len(built.assumptions) == 2


def test_requirements_of_none_means_an_empty_list(plan):
    sections = working_sections()
    sections["requirements"] = "none"
    assert codegen.build(plan, sections).requirements == []


def test_a_malformed_response_gets_one_format_repair_not_a_code_attempt(plan):
    model = FakeModel(["I am afraid I cannot do that.", working_sections()])
    sections = codegen.call_model(
        model, "prompt", field_names=prompts.EXPERIMENT_FIELD_NAMES, max_format_retries=1
    )
    assert model.calls == 2
    assert sections["load_data"].startswith("def load_data")
    assert "could not be parsed" in model.prompts[1]


def test_a_response_that_never_parses_raises_rather_than_running_garbage(plan):
    model = FakeModel(["nope", "still nope", "nope again"])
    try:
        codegen.call_model(
            model, "prompt", field_names=prompts.EXPERIMENT_FIELD_NAMES, max_format_retries=1
        )
    except SectionParseError as exc:
        assert exc.missing
    else:
        raise AssertionError("an unparseable response must raise, not return partial sections")


def test_the_prompt_shows_the_exact_format_the_parser_accepts(plan):
    prompt = prompts.codegen_prompt(
        plan, data_block="D", available_packages=["numpy"], gpus="1", timeout_seconds=600
    )
    for name in prompts.EXPERIMENT_FIELD_NAMES:
        assert f"===BEGIN {name}===" in prompt
    assert "Wall-clock budget: 600 seconds" in prompt


def test_the_fix_prompt_escalates_after_a_repeat(plan):
    ordinary = prompts.fix_prompt(
        plan, code="x", diagnosis_summary="boom", failure_class="runtime_logic", evidence="e",
        data_block="d", available_packages=[], gpus="", timeout_seconds=60, attempt=1, escalate=False,
    )
    escalated = prompts.fix_prompt(
        plan, code="x", diagnosis_summary="boom", failure_class="runtime_logic", evidence="e",
        data_block="d", available_packages=[], gpus="", timeout_seconds=60, attempt=2, escalate=True,
    )
    assert "SECOND time" not in ordinary
    assert "different implementation strategy" in escalated
