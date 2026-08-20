"""Environment provisioning, including the Barkla-specific placement rules."""

from __future__ import annotations

from pathlib import Path

from coder_agent import envman


def test_import_names_map_to_the_distributions_that_provide_them():
    assert envman.resolve_package("sklearn") == "scikit-learn"
    assert envman.resolve_package("cv2") == "opencv-python-headless"
    assert envman.resolve_package("PIL") == "pillow"
    assert envman.resolve_package("sklearn.linear_model") == "scikit-learn"
    assert envman.resolve_package("pandas") == "pandas", "unmapped names pass through"


def test_dead_imports_are_not_treated_as_installable_aliases():
    """The distinction that job 10274056 was lost on.

    `sklearn` -> `scikit-learn` is an alias: installing the distribution really
    does provide the import. `pymc3` -> `pymc` is a successor: PyMC 3 is
    end-of-life and installing PyMC 5 provides no `pymc3` module at all. Routing
    the second like the first gives an install that reports success while the
    identical ImportError returns on every retry.
    """
    assert envman.is_dead_import("sklearn") is None, "an alias is installable"

    for module, expected in [("pymc3", "pymc"), ("pystan", "cmdstanpy"), ("theano", "pytensor")]:
        dead = envman.is_dead_import(module)
        assert dead is not None, f"{module} cannot be satisfied by any install"
        assert dead[0] == expected
        assert dead[1], "a replacement must carry usable guidance for the code generator"


def test_a_dead_import_is_detected_through_a_submodule():
    assert envman.is_dead_import("pymc3.distributions") is not None


def test_the_denylist_refuses_packages_that_would_fight_the_agent(tmp_path: Path):
    env = envman.ExperimentEnv(venv_path=tmp_path, python_bin="/usr/bin/python3")
    ok, detail = envman.install(env, ["vllm"], reason="test")

    assert ok is False
    assert "denylist" in detail
    assert env.installs[0]["ok"] is False


def test_an_offline_node_reports_a_useful_fix_instead_of_retrying(tmp_path: Path, monkeypatch):
    """Barkla compute nodes are frequently isolated; the message must say what to do."""
    monkeypatch.setattr(envman, "has_network", lambda *a, **k: False)
    env = envman.ExperimentEnv(venv_path=tmp_path, python_bin="/usr/bin/python3")

    ok, detail = envman.install(env, ["geopandas"], reason="test")

    assert ok is False
    assert "no outbound network" in detail
    assert "build_base_env.sh" in detail


def test_the_venv_lands_beside_the_results_by_default(tmp_path: Path):
    experiment_dir = tmp_path / "H1"
    experiment_dir.mkdir()
    env = envman.provision(experiment_dir)

    assert env.venv_path == experiment_dir / ".venv"
    assert Path(env.python_bin).exists()


def test_venv_root_moves_the_venv_off_the_quota_bearing_filesystem(tmp_path: Path):
    """On Barkla this is localscratch: no inode quota, node-local, disposable."""
    experiment_dir = tmp_path / "results" / "H1"
    experiment_dir.mkdir(parents=True)
    localscratch = tmp_path / "localscratch"

    env = envman.provision(experiment_dir, venv_root=str(localscratch))

    assert env.venv_path == localscratch / "H1" / ".venv"
    assert Path(env.python_bin).exists()
    assert not (experiment_dir / ".venv").exists(), "results dir stays free of venv files"


def test_a_base_env_without_an_interpreter_fails_loudly(tmp_path: Path):
    experiment_dir = tmp_path / "H1"
    experiment_dir.mkdir()
    empty = tmp_path / "not-an-env"
    empty.mkdir()

    try:
        envman.provision(experiment_dir, base_env=str(empty))
    except envman.EnvError as exc:
        assert "build_base_env.sh" in str(exc)
    else:
        raise AssertionError("a misconfigured CODER_BASE_ENV must not fail silently")


def test_availability_is_asked_of_the_target_interpreter(tmp_path: Path):
    experiment_dir = tmp_path / "H1"
    experiment_dir.mkdir()
    env = envman.provision(experiment_dir)

    assert envman.is_available(env, "json") is True
    assert envman.is_available(env, "a_module_that_does_not_exist") is False
