"""The classifier, against real traceback text.

The first test in this file is the one that matters most: it is the literal
failure recorded in coder_agent_summary_20260819T172608Z.json, where three fix
attempts were spent regenerating code in response to a missing package.
"""

from __future__ import annotations

from coder_agent import diagnose

PANDAS_TRACEBACK = """Traceback (most recent call last):
  File "/mnt/fastscratch/users/sgyshere/multi-agent-langraph/experiments/H1/run.py", line 30, in <module>
    import pandas as pd
ModuleNotFoundError: No module named 'pandas'"""


def test_missing_package_routes_to_the_environment_not_the_code_generator():
    d = diagnose.classify(exit_code=1, stdout="", stderr=PANDAS_TRACEBACK)

    assert d is not None
    assert d.failure_class == "missing_dependency"
    assert d.route == diagnose.ROUTE_ENV
    assert d.module == "pandas"
    assert d.route != diagnose.ROUTE_CODE  # the whole point


def test_the_same_failure_has_the_same_signature_even_at_a_different_line():
    first = diagnose.classify(exit_code=1, stdout="", stderr=PANDAS_TRACEBACK)
    shifted = diagnose.classify(
        exit_code=1, stdout="", stderr=PANDAS_TRACEBACK.replace("line 30", "line 47")
    )
    assert first.signature == shifted.signature


def test_different_missing_packages_have_different_signatures():
    a = diagnose.classify(exit_code=1, stdout="", stderr=PANDAS_TRACEBACK)
    b = diagnose.classify(
        exit_code=1, stdout="", stderr=PANDAS_TRACEBACK.replace("pandas", "geopandas")
    )
    assert a.signature != b.signature


def test_missing_shared_library_is_terminal_because_pip_cannot_fix_it():
    d = diagnose.classify(
        exit_code=1,
        stdout="",
        stderr="OSError: libcudart.so.12: cannot open shared object file: No such file or directory",
    )
    assert d.failure_class == "missing_system_library"
    assert d.route == diagnose.ROUTE_TERMINAL


def test_installed_but_wrong_version_is_a_code_problem_not_an_install():
    d = diagnose.classify(
        exit_code=1,
        stdout="",
        stderr=(
            "Traceback (most recent call last):\n"
            '  File "run.py", line 12, in <module>\n'
            "    from pymc3 import Model\n"
            "ImportError: cannot import name 'Model' from 'pymc3'"
        ),
    )
    assert d.failure_class == "api_mismatch"
    assert d.route == diagnose.ROUTE_CODE


def test_missing_data_file_routes_to_the_data_layer():
    d = diagnose.classify(
        exit_code=1,
        stdout="",
        stderr=(
            "Traceback (most recent call last):\n"
            '  File "run.py", line 42, in load_data\n'
            "    df = pd.read_csv(path)\n"
            "FileNotFoundError: [Errno 2] No such file or directory: 'data/aqs_pm25.csv'"
        ),
    )
    assert d.failure_class == "missing_data"
    assert d.route == diagnose.ROUTE_DATA
    assert d.path == "data/aqs_pm25.csv"


def test_a_missing_non_data_file_is_a_code_bug_not_a_data_problem():
    d = diagnose.classify(
        exit_code=1,
        stdout="",
        stderr=(
            "Traceback (most recent call last):\n"
            '  File "run.py", line 8, in main\n'
            "FileNotFoundError: [Errno 2] No such file or directory: '/opt/stan/bin/stanc'"
        ),
    )
    assert d.route == diagnose.ROUTE_CODE


def test_oom_and_timeout_route_to_downscale():
    oom = diagnose.classify(
        exit_code=1, stdout="", stderr="torch.cuda.OutOfMemoryError: CUDA out of memory."
    )
    assert oom.route == diagnose.ROUTE_DOWNSCALE

    timeout = diagnose.classify(exit_code=-9, stdout="sampling", stderr="", timed_out=True)
    assert timeout.failure_class == "resource"
    assert timeout.route == diagnose.ROUTE_DOWNSCALE
    assert timeout.details["kind"] == "timeout"


def test_syntax_error_is_reported_as_syntax():
    d = diagnose.classify(
        exit_code=1, stdout="", stderr='  File "run.py", line 9\n    def f(:\nSyntaxError: invalid syntax'
    )
    assert d.failure_class == "syntax"


def test_ordinary_runtime_failure_falls_through_to_code_regeneration():
    d = diagnose.classify(
        exit_code=1,
        stdout="",
        stderr=(
            "Traceback (most recent call last):\n"
            '  File "run.py", line 141, in evaluate\n'
            "IndexError: index 5 is out of bounds for axis 0 with size 3"
        ),
    )
    assert d.failure_class == "runtime_logic"
    assert d.route == diagnose.ROUTE_CODE


def test_a_clean_run_that_wrote_nothing_is_still_a_failure():
    d = diagnose.classify(exit_code=0, stdout="all done\n", stderr="", results_present=False)
    assert d.failure_class == "contract"
    assert d.details["problem"] == "missing_results_file"


def test_all_zero_metrics_are_rejected_as_degenerate():
    d = diagnose.classify(
        exit_code=0,
        stdout="",
        stderr="",
        results_present=True,
        results={"metrics": {"beta_interaction": 0.0, "beta_greenspace": None}},
    )
    assert d.failure_class == "contract"
    assert d.details["problem"] == "degenerate_metrics"


def test_partially_zero_metrics_are_accepted_because_zero_can_be_a_real_answer():
    d = diagnose.classify(
        exit_code=0,
        stdout="",
        stderr="",
        results_present=True,
        results={"metrics": {"beta_interaction": 0.0, "beta_greenspace": -0.31}},
    )
    assert d is None


def test_a_healthy_run_produces_no_diagnosis():
    d = diagnose.classify(
        exit_code=0,
        stdout="done",
        stderr="",
        results_present=True,
        results={"metrics": {"pm25_deprivation_interaction": 0.42}},
        expected_metrics=["Posterior mean of the interaction coefficient between PM2.5 and deprivation"],
    )
    assert d is None


def test_ordinary_stdout_is_not_mistaken_for_a_traceback():
    # A run whose stdout merely mentions progress must not be classified as an
    # error just because the classifier looked at the tail of the log.
    d = diagnose.classify(
        exit_code=0,
        stdout="chain 1 complete\nchain 2 complete\n",
        stderr="",
        results_present=True,
        results={"metrics": {"x": 1.0}},
    )
    assert d is None


def test_the_last_traceback_wins_when_a_run_logged_a_handled_one_earlier():
    log = (
        "Traceback (most recent call last):\n"
        "  File \"run.py\", line 5, in fetch\n"
        "ConnectionError: transient\n"
        "retrying...\n"
        "Traceback (most recent call last):\n"
        "  File \"run.py\", line 88, in evaluate\n"
        "ValueError: shapes do not align"
    )
    d = diagnose.classify(exit_code=1, stdout="", stderr=log)
    assert "shapes do not align" in d.summary
