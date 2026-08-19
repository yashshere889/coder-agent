"""Shared fixtures. No test here touches the network, a real model, or a cluster."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from coder_agent import prompts
from coder_agent.config import settings as real_settings
from coder_agent.sections import render_sections

PLAN_FIXTURE = {
    "experiment_plans": [
        {
            "hypothesis_id": "H1",
            "feasible": True,
            "objective": "Test whether neighbourhood PM2.5 predicts cardiovascular admissions.",
            "design": "Spatial longitudinal cohort with Bayesian hierarchical modelling.",
            "variables": {
                "independent": ["PM2.5 concentration", "Neighborhood deprivation index"],
                "dependent": ["Cardiovascular hospital admission rate"],
            },
            "data_requirements": {
                "source": "EPA Air Quality System for PM2.5; CMS Medicare Hospital Claims for admissions",
                "description": "Neighbourhood-level panel linking monitors to census tracts",
                "preprocessing_steps": ["Aggregate monitors to tract level", "Standardize covariates"],
            },
            "methods": [{"name": "Bayesian hierarchical model", "description": "CAR spatial effects"}],
            "evaluation": {
                "metrics": [
                    "Posterior mean of the interaction coefficient between PM2.5 and deprivation",
                    "Posterior mean of the interaction coefficient between PM2.5 and greenspace",
                ],
                "baseline": "No spatial variation in the exposure-outcome relationship.",
                "success_criteria": "The deprivation interaction is positive with a 95% CI excluding zero.",
                "refute_criteria": "Neither interaction differs from zero.",
            },
            "implementation_steps": [
                {"step": 2, "description": "Build the spatial weights matrix"},
                {"step": 1, "description": "Load the panel dataset"},
            ],
            "estimated_complexity": "medium",
            "risks": ["Exposure misclassification at the neighbourhood level"],
        }
    ],
    "priority_order": [{"hypothesis_id": "H1", "rank": 1}],
}


@pytest.fixture
def plan_file(tmp_path: Path) -> Path:
    import json

    path = tmp_path / "experiment_plan.json"
    path.write_text(json.dumps(PLAN_FIXTURE))
    return path


@pytest.fixture
def plan(plan_file: Path):
    from coder_agent.plan import load_plans

    return load_plans(plan_file)[0]


@pytest.fixture
def config(tmp_path: Path):
    """Real Settings with the budgets and timeouts a test can afford."""
    return dataclasses.replace(
        real_settings,
        experiments_dir=str(tmp_path / "experiments"),
        base_env="",
        experiment_gpus="",
        max_code_attempts=3,
        max_env_repairs=3,
        max_format_retries=1,
        timeout_low=60,
        timeout_medium=60,
        timeout_high=60,
        memory_limit_gb=0,
    )


def working_sections(body: str = "return {'x': 1.5}", imports: str = "import math") -> dict[str, str]:
    """A complete, runnable set of sections. Mutate this rather than rebuilding it."""
    return {
        "imports": imports,
        "configuration": "SEED = 7",
        "helpers": "def _noop():\n    return None",
        "load_data": "def load_data():\n    return [1, 2, 3]",
        "preprocess": "def preprocess(raw):\n    return list(raw)",
        "build_model": "def build_model(prepared):\n    return {'kind': 'stub'}",
        "run_experiment": f"def run_experiment(model, prepared):\n    {body}",
        "evaluate": (
            "def evaluate(outputs, prepared):\n"
            "    return {'metrics': {'pm25_deprivation_interaction': outputs['x'],\n"
            "                        'pm25_greenspace_interaction': -0.4},\n"
            "            'hypothesis_outcome': 'supported',\n"
            "            'meets_success_criteria': True,\n"
            "            'notes': 'stub'}"
        ),
        "requirements": "none",
        "assumptions": "Data is synthetic for this test.",
    }


class FakeModel:
    """Returns queued responses in order; the last one repeats.

    Counting calls is how the regression tests assert the important negative:
    that an environment failure produced *no* codegen call at all.
    """

    def __init__(self, responses: list[dict[str, str] | str]):
        self.responses = responses
        self.prompts: list[str] = []

    @property
    def calls(self) -> int:
        return len(self.prompts)

    def complete(self, prompt: str, system: str = "", attempts: int = 3) -> str:
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self.responses) - 1)
        response = self.responses[index]
        return response if isinstance(response, str) else render_sections(response)

    def health(self):
        return True, "fake"


@pytest.fixture
def field_names():
    return prompts.EXPERIMENT_FIELD_NAMES
