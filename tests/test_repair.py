"""The repair router: does each failure reach the thing that can fix it?"""

from __future__ import annotations

from pathlib import Path

import pytest

from coder_agent import diagnose, envman, repair

PANDAS_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "run.py", line 30, in <module>\n'
    "    import pandas as pd\n"
    "ModuleNotFoundError: No module named 'pandas'"
)


@pytest.fixture
def env(tmp_path: Path):
    return envman.ExperimentEnv(venv_path=tmp_path / ".venv", python_bin="/usr/bin/python3")


def test_a_missing_package_installs_and_does_not_regenerate(env, monkeypatch):
    """The regression for the observed failure: env repair must not touch the code."""
    installed: list[list[str]] = []

    def fake_install(target_env, packages, *, reason):
        installed.append(packages)
        target_env.record(packages, ok=True, detail="ok", reason=reason)
        return True, "installed"

    monkeypatch.setattr(envman, "install", fake_install)

    d = diagnose.classify(exit_code=1, stdout="", stderr=PANDAS_TRACEBACK)
    outcome = repair.repair(d, env=env, code="import pandas", tracker=repair.ProgressTracker())

    assert outcome.action == "install"
    assert outcome.ok
    assert outcome.regenerate is False       # the code was never the problem
    assert outcome.code is None              # and it was not rewritten
    assert installed == [["pandas"]]


def test_a_wrong_import_to_package_guess_falls_back_to_the_import_name(env, monkeypatch):
    attempts: list[list[str]] = []

    def fake_install(target_env, packages, *, reason):
        attempts.append(packages)
        ok = packages != ["scikit-learn"]  # pretend the mapped name is not on the index
        target_env.record(packages, ok=ok, detail="", reason=reason)
        return ok, "no matching distribution" if not ok else "installed"

    monkeypatch.setattr(envman, "install", fake_install)

    d = diagnose.classify(
        exit_code=1, stdout="", stderr=PANDAS_TRACEBACK.replace("pandas", "sklearn")
    )
    outcome = repair.repair(d, env=env, code="import sklearn", tracker=repair.ProgressTracker())

    assert attempts == [["scikit-learn"], ["sklearn"]]
    assert outcome.ok


def test_an_install_that_cannot_succeed_stops_rather_than_looping(env, monkeypatch):
    monkeypatch.setattr(
        envman, "install", lambda e, p, *, reason: (False, "no outbound network from this node")
    )
    d = diagnose.classify(exit_code=1, stdout="", stderr=PANDAS_TRACEBACK)
    outcome = repair.repair(d, env=env, code="", tracker=repair.ProgressTracker())

    assert outcome.terminal
    assert "build_base_env" in " ".join(outcome.notes)


def test_the_same_failure_three_times_stops_the_run(env, monkeypatch):
    monkeypatch.setattr(envman, "install", lambda e, p, *, reason: (True, "installed"))
    tracker = repair.ProgressTracker()
    d = diagnose.classify(exit_code=1, stdout="", stderr=PANDAS_TRACEBACK)

    first = repair.repair(d, env=env, code="", tracker=tracker)
    second = repair.repair(d, env=env, code="", tracker=tracker)
    third = repair.repair(d, env=env, code="", tracker=tracker)

    assert first.ok and not first.terminal
    assert second.ok and not second.terminal
    assert third.terminal, "a third identical failure must stop, not consume more budget"
    assert "no-progress guard" in " ".join(third.notes)


def test_a_repeated_code_failure_escalates_to_a_different_approach(env):
    tracker = repair.ProgressTracker()
    d = diagnose.classify(
        exit_code=1,
        stdout="",
        stderr='Traceback (most recent call last):\n  File "run.py", line 4, in evaluate\nValueError: bad',
    )
    first = repair.repair(d, env=env, code="x", tracker=tracker)
    second = repair.repair(d, env=env, code="x", tracker=tracker)

    assert first.regenerate and not first.escalate
    assert second.regenerate and second.escalate


def test_an_oom_is_shrunk_deterministically_without_a_model_call(env):
    code = "draws = 4000\nchains = 8\nbatch_size = 256\nYEAR = 2020\n"
    d = diagnose.classify(exit_code=1, stdout="", stderr="CUDA out of memory")
    outcome = repair.repair(d, env=env, code=code, tracker=repair.ProgressTracker())

    assert outcome.action == "downscale"
    assert outcome.regenerate is False
    assert "draws = 2000" in outcome.code
    assert "chains = 4" in outcome.code
    assert "YEAR = 2020" in outcome.code, "a year is not a cost knob and must be left alone"


def test_downscale_respects_its_floors():
    code, changes = repair.downscale("chains = 2\ndraws = 250\n")
    assert changes == []
    assert code == "chains = 2\ndraws = 250\n"


def test_an_oom_with_nothing_to_shrink_asks_for_a_different_approach(env):
    d = diagnose.classify(exit_code=1, stdout="", stderr="CUDA out of memory")
    outcome = repair.repair(
        d, env=env, code="model.fit(everything)", tracker=repair.ProgressTracker()
    )
    assert outcome.regenerate and outcome.escalate


def test_snapshots_keep_every_failed_attempt(tmp_path: Path):
    d = diagnose.classify(exit_code=1, stdout="", stderr=PANDAS_TRACEBACK)
    target = repair.snapshot(tmp_path, 2, "import pandas\n", d)

    assert (target / "run.py").read_text() == "import pandas\n"
    assert "missing_dependency" in (target / "diagnosis.json").read_text()
    assert "ModuleNotFoundError" in (target / "evidence.txt").read_text()
