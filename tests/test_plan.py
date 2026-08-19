"""Parsing the plan file — the agent's entire input contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coder_agent.plan import PlanError, load_plans, parse_plan

from .conftest import PLAN_FIXTURE


def test_the_real_plan_file_shape_parses(plan):
    assert plan.hypothesis_id == "H1"
    assert plan.complexity == "medium"
    assert len(plan.evaluation.metrics) == 2
    assert "Aggregate monitors to tract level" in plan.data_requirements.preprocessing_steps


def test_implementation_steps_are_sorted_by_their_own_step_number(plan):
    # The fixture lists step 2 before step 1 on purpose.
    assert plan.implementation_steps[0] == "Load the panel dataset"
    assert plan.implementation_steps[1] == "Build the spatial weights matrix"


def test_implementation_steps_also_accept_a_plain_list_of_strings():
    document = json.loads(json.dumps(PLAN_FIXTURE))
    document["experiment_plans"][0]["implementation_steps"] = ["first", "second"]
    parsed = parse_plan(document["experiment_plans"][0])
    assert parsed.implementation_steps == ("first", "second")


def test_a_plan_with_no_metrics_is_rejected():
    document = json.loads(json.dumps(PLAN_FIXTURE))
    document["experiment_plans"][0]["evaluation"]["metrics"] = []
    with pytest.raises(PlanError, match="nothing to check it against"):
        parse_plan(document["experiment_plans"][0])


def test_a_plan_with_no_objective_is_rejected():
    document = json.loads(json.dumps(PLAN_FIXTURE))
    document["experiment_plans"][0]["objective"] = "   "
    with pytest.raises(PlanError, match="objective"):
        parse_plan(document["experiment_plans"][0])


def test_infeasible_plans_are_not_run(tmp_path: Path):
    document = json.loads(json.dumps(PLAN_FIXTURE))
    document["experiment_plans"][0]["feasible"] = False
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(document))

    with pytest.raises(PlanError, match="feasible=false"):
        load_plans(path)


def test_plans_come_back_in_the_planners_priority_order(tmp_path: Path):
    document = json.loads(json.dumps(PLAN_FIXTURE))
    second = json.loads(json.dumps(document["experiment_plans"][0]))
    second["hypothesis_id"] = "H2"
    document["experiment_plans"].append(second)
    document["priority_order"] = [
        {"hypothesis_id": "H2", "rank": 1},
        {"hypothesis_id": "H1", "rank": 2},
    ]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(document))

    assert [p.hypothesis_id for p in load_plans(path)] == ["H2", "H1"]


def test_selecting_an_unknown_hypothesis_id_is_an_error(plan_file: Path):
    with pytest.raises(PlanError, match="H9"):
        load_plans(plan_file, hypothesis_ids=["H9"])


def test_a_hostile_hypothesis_id_cannot_escape_the_experiments_directory():
    document = json.loads(json.dumps(PLAN_FIXTURE))
    document["experiment_plans"][0]["hypothesis_id"] = "../../etc/passwd"
    parsed = parse_plan(document["experiment_plans"][0])
    assert "/" not in parsed.slug and ".." not in parsed.slug


def test_a_missing_file_says_so_plainly(tmp_path: Path):
    with pytest.raises(PlanError, match="not found"):
        load_plans(tmp_path / "nope.json")


def test_malformed_json_says_so_plainly(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(PlanError, match="not valid JSON"):
        load_plans(path)
