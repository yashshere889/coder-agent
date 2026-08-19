"""End-to-end: a plan file in, a real subprocess run out.

These tests create real venvs and execute real generated code. What they never
do is reach a model or the network — the model is a `FakeModel` with canned
sections, and every install is intercepted.

The first test is the reason this project exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coder_agent import envman, loop

from .conftest import FakeModel, working_sections


@pytest.fixture
def experiments_root(tmp_path: Path) -> Path:
    return tmp_path / "experiments"


def _stub_installer(experiment_dir: Path, *, fail_first: bool = True):
    """An installer that satisfies an import by dropping a stub module next to run.py.

    Standing in for pip: `fail_first` models the realistic case where the
    pre-flight guess does not resolve, so the reactive repair — which knows the
    exact missing module name — is what actually fixes it.
    """
    calls: list[list[str]] = []

    def install(env, packages, *, reason):
        calls.append(list(packages))
        if fail_first and len(calls) == 1:
            env.record(packages, ok=False, detail="no matching distribution", reason=reason)
            return False, "no matching distribution found"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        for package in packages:
            module = package.replace("-", "_")
            (experiment_dir / f"{module}.py").write_text("VALUE = 1.5\n")
        env.record(packages, ok=True, detail="installed", reason=reason)
        return True, "installed"

    return install, calls


def test_a_missing_package_is_installed_and_the_code_is_never_regenerated(
    plan, config, experiments_root, monkeypatch
):
    """The regression for coder_agent_summary_20260819T172608Z.json.

    That run spent all three fix attempts asking the model to rewrite an
    experiment because `pandas` was not installed. Here the equivalent failure
    must cost exactly one install and zero regenerations.
    """
    experiment_dir = experiments_root / "H1"
    install, install_calls = _stub_installer(experiment_dir)
    monkeypatch.setattr(envman, "install", install)

    sections = working_sections(
        imports="import fictional_stats_pkg",
        body="return {'x': fictional_stats_pkg.VALUE}",
    )
    model = FakeModel([sections])

    report = loop.run_plan(plan, model, config=config, experiments_root=experiments_root)

    assert report.status == loop.STATUS_COMPLETED
    assert model.calls == 1, "the model was asked to write code once; the rest was environment work"
    assert report.code_attempts_used == 1

    env_failures = [
        a for a in report.attempts if (a["diagnosis"] or {}).get("failure_class") == "missing_dependency"
    ]
    assert len(env_failures) == 1
    assert env_failures[0]["diagnosis"]["route"] == "env_repair"
    assert env_failures[0]["repair"]["action"] == "install"
    assert env_failures[0]["repair"]["regenerate"] is False
    assert ["fictional_stats_pkg"] in install_calls
    assert report.results["metrics"]["pm25_deprivation_interaction"] == 1.5


def test_a_clean_plan_runs_on_the_first_attempt(plan, config, experiments_root):
    model = FakeModel([working_sections()])
    report = loop.run_plan(plan, model, config=config, experiments_root=experiments_root)

    assert report.status == loop.STATUS_COMPLETED
    assert model.calls == 1
    assert len(report.attempts) == 1
    assert report.results["status"] == "completed"
    assert (Path(report.experiment_dir) / "run.py").exists()
    assert (Path(report.experiment_dir) / "run_report.json").exists()
    assert (Path(report.experiment_dir) / "environment.json").exists()


def test_a_real_code_bug_is_regenerated_with_the_traceback_quoted_back(
    plan, config, experiments_root
):
    broken = working_sections(body="return {'x': 1 / 0}")
    model = FakeModel([broken, working_sections()])

    report = loop.run_plan(plan, model, config=config, experiments_root=experiments_root)

    assert report.status == loop.STATUS_COMPLETED
    assert model.calls == 2, "one generation plus one regeneration"
    assert report.attempts[0]["diagnosis"]["failure_class"] == "runtime_logic"
    assert report.attempts[0]["repair"]["regenerate"] is True
    assert "ZeroDivisionError" in model.prompts[1]
    assert (Path(report.experiment_dir) / "attempts" / "attempt_1" / "run.py").exists()


def test_a_syntax_error_never_reaches_a_subprocess(plan, config, experiments_root):
    broken = working_sections()
    broken["helpers"] = "def broken(:\n    pass"
    model = FakeModel([broken, working_sections()])

    report = loop.run_plan(plan, model, config=config, experiments_root=experiments_root)

    assert report.attempts[0]["stage"] == "preflight"
    assert report.attempts[0]["diagnosis"]["failure_class"] == "syntax"
    assert report.status == loop.STATUS_COMPLETED


def test_dangerous_code_is_refused_before_it_runs(plan, config, experiments_root):
    dangerous = working_sections()
    dangerous["helpers"] = "import os\ndef wipe():\n    os.system('rm -rf /tmp/whatever')"
    model = FakeModel([dangerous, working_sections()])

    report = loop.run_plan(plan, model, config=config, experiments_root=experiments_root)

    assert report.attempts[0]["diagnosis"]["failure_class"] == "safety"
    assert report.attempts[0]["stage"] == "preflight"
    assert report.status == loop.STATUS_COMPLETED


def test_a_run_that_writes_no_metrics_is_not_accepted_as_success(plan, config, experiments_root):
    hollow = working_sections()
    hollow["evaluate"] = "def evaluate(outputs, prepared):\n    return {'metrics': {}}"
    model = FakeModel([hollow, working_sections()])

    report = loop.run_plan(plan, model, config=config, experiments_root=experiments_root)

    assert report.attempts[0]["diagnosis"]["failure_class"] == "contract"
    assert report.attempts[0]["diagnosis"]["details"]["problem"] == "empty_metrics"
    assert report.status == loop.STATUS_COMPLETED


def test_an_experiment_that_keeps_failing_identically_stops_early(plan, config, experiments_root):
    """The no-progress guard fires *before* the budget runs out, and says so.

    A model that returns the same broken code every time produces the same
    diagnosis signature every time. Spending the rest of the budget on it is
    exactly the behaviour this agent was built to avoid, so the run stops at the
    third identical failure with a status that names the reason.
    """
    model = FakeModel([working_sections(body="return {'x': 1 / 0}")])  # always broken

    report = loop.run_plan(plan, model, config=config, experiments_root=experiments_root)

    assert report.status == loop.STATUS_NO_PROGRESS
    assert "same failure has now occurred 3 times" in report.reason
    signatures = {a["diagnosis"]["signature"] for a in report.attempts}
    assert len(signatures) == 1, "every attempt failed the same way, which is why it stopped"


def test_the_code_budget_stops_an_experiment_whose_failures_keep_changing(
    plan, config, experiments_root
):
    """A different bug each time never trips the no-progress guard, so the budget is what stops it."""
    model = FakeModel(
        [
            working_sections(body="return {'x': 1 / 0}"),
            working_sections(body="return {'x': undefined_name}"),
            working_sections(body="return {'x': [][5]}"),
            working_sections(body="return {'x': int('not a number')}"),
        ]
    )

    report = loop.run_plan(plan, model, config=config, experiments_root=experiments_root)

    assert report.status == loop.STATUS_BUDGET_EXHAUSTED
    assert report.code_attempts_used == config.max_code_attempts
    assert "code attempts exhausted" in report.reason


def test_a_surrogate_input_withholds_the_hypothesis_verdict(plan, config, experiments_root):
    """The fixture plan needs CMS claims, which cannot be obtained openly."""
    model = FakeModel([working_sections()])
    report = loop.run_plan(plan, model, config=config, experiments_root=experiments_root)

    assert report.status == loop.STATUS_COMPLETED
    assert report.results["hypothesis_outcome"] == "supported"  # what the code claimed
    assert report.evaluation["hypothesis_outcome"] == "not_assessable"  # what the agent reports
    assert report.evaluation["meets_success_criteria"] is False
    assert "NOT interpretable" in report.evaluation["methodological_validity"]

    provenance = json.loads((Path(report.experiment_dir) / "data_provenance.json").read_text())
    assert provenance["all_inputs_real"] is False
    assert any("Data Use Agreement" in i["reason"] for i in provenance["inputs"])


def test_staged_real_data_restores_a_real_verdict(plan, config, experiments_root, tmp_path):
    staging = tmp_path / "staged"
    staging.mkdir()
    for name in (
        "epa_air_quality_system_pm25_monitors.csv",
        "cms_medicare_hospital_claims_admissions.csv",
        "census_american_community_survey_deprivation.csv",
        "neighbourhood_level_panel_linking_monitors_census_tracts.csv",
    ):
        (staging / name).write_text("a,b\n1,2\n")

    model = FakeModel([working_sections()])
    report = loop.run_plan(
        plan, model, config=config, experiments_root=experiments_root, staging_dir=staging
    )

    provenance = json.loads((Path(report.experiment_dir) / "data_provenance.json").read_text())
    assert provenance["all_inputs_real"] is True
    assert report.evaluation["hypothesis_outcome"] == "supported"


def test_the_generated_code_is_told_which_inputs_are_synthetic(plan, config, experiments_root):
    model = FakeModel([working_sections()])
    loop.run_plan(plan, model, config=config, experiments_root=experiments_root)

    assert "SURROGATE" in model.prompts[0]
    assert "synthesize_" in model.prompts[0]


def test_a_run_summary_covers_every_plan(plan, config, experiments_root):
    model = FakeModel([working_sections()])
    summary = loop.run_all([plan], model, config=config, experiments_root=experiments_root)

    assert summary["total"] == 1 and summary["completed"] == 1
    assert (experiments_root / "summary.json").exists()


def test_one_plan_failing_does_not_stop_the_others(plan, config, experiments_root, monkeypatch):
    import dataclasses

    other = dataclasses.replace(plan, hypothesis_id="H2")
    calls = {"n": 0}
    original = loop.run_plan

    def flaky(target_plan, *args, **kwargs):
        calls["n"] += 1
        if target_plan.hypothesis_id == "H1":
            raise RuntimeError("harness exploded")
        return original(target_plan, *args, **kwargs)

    monkeypatch.setattr(loop, "run_plan", flaky)
    summary = loop.run_all(
        [plan, other], FakeModel([working_sections()]), config=config, experiments_root=experiments_root
    )

    assert calls["n"] == 2
    assert summary["completed"] == 1
    assert "harness error" in summary["experiments"][0]["reason"]
