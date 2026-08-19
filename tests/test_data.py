"""The provenance gate: synthetic data must never be reported as evidence."""

from __future__ import annotations

from pathlib import Path

from coder_agent import data


def test_restricted_sources_become_labelled_surrogates_not_silent_downloads():
    sources = data.resolve(["CMS Medicare Hospital Claims for cardiovascular admissions"], network=True)

    assert len(sources) == 1
    assert sources[0].kind == data.KIND_SURROGATE
    assert "Data Use Agreement" in sources[0].reason


def test_an_open_source_is_real_when_the_node_has_network():
    sources = data.resolve(["EPA Air Quality System PM2.5 monitor data"], network=True)
    assert sources[0].kind == data.KIND_REAL_DOWNLOAD
    assert "aqs.epa.gov" in sources[0].uri


def test_the_same_open_source_is_a_surrogate_when_the_node_is_isolated():
    sources = data.resolve(["EPA Air Quality System PM2.5 monitor data"], network=False)
    assert sources[0].kind == data.KIND_SURROGATE
    assert "no outbound network" in sources[0].reason


def test_a_staged_file_beats_everything_including_a_restricted_source(tmp_path: Path):
    staging = tmp_path / "staged"
    staging.mkdir()
    (staging / "medicare_claims_2020.csv").write_text("beneficiary,admissions\n1,3\n")

    sources = data.resolve(["CMS Medicare claims for admissions"], staging_dir=staging, network=True)

    assert sources[0].kind == data.KIND_REAL_LOCAL
    assert sources[0].local_path.endswith("medicare_claims_2020.csv")
    assert sources[0].checksum


def test_one_surrogate_makes_the_whole_run_non_evidential():
    sources = [
        data.DataSource(name="a", kind=data.KIND_REAL_LOCAL),
        data.DataSource(name="b", kind=data.KIND_SURROGATE),
    ]
    assert data.all_real(sources) is False
    assert "NOT interpretable" in data.verdict(sources)


def test_the_hypothesis_verdict_is_withheld_on_surrogate_data():
    """The model may claim `supported`; on synthetic inputs that claim is discarded."""
    sources = [data.DataSource(name="cms", kind=data.KIND_SURROGATE)]
    results = {
        "metrics": {"beta": 0.42},
        "hypothesis_outcome": "supported",
        "meets_success_criteria": True,
    }

    evaluation = data.stamp_evaluation(
        results, sources, success_criteria="CI excludes zero", refute_criteria="it does not"
    )

    assert evaluation["hypothesis_outcome"] == "not_assessable"
    assert evaluation["meets_success_criteria"] is False
    assert evaluation["model_reported_outcome_discarded"] == "supported"
    assert "synthetic" in evaluation["withheld_because"].lower()
    assert evaluation["metrics"] == {"beta": 0.42}, "the numbers are still reported, just not as evidence"


def test_the_verdict_stands_when_every_input_is_real():
    sources = [data.DataSource(name="epa", kind=data.KIND_REAL_DOWNLOAD)]
    evaluation = data.stamp_evaluation(
        {"metrics": {"beta": 0.42}, "hypothesis_outcome": "supported", "meets_success_criteria": True},
        sources,
        success_criteria="CI excludes zero",
        refute_criteria="it does not",
    )
    assert evaluation["hypothesis_outcome"] == "supported"
    assert evaluation["meets_success_criteria"] is True


def test_the_prompt_tells_the_model_which_inputs_are_synthetic():
    sources = data.resolve(
        ["EPA Air Quality System PM2.5", "CMS Medicare claims"], network=True
    )
    block = data.prompt_block(sources)

    assert "REAL, fetch from" in block
    assert "SURROGATE" in block
    assert "synthesize_" in block
    assert "never assume an unstated file already exists" in block


def test_provenance_is_written_next_to_the_results(tmp_path: Path):
    sources = data.resolve(["CMS Medicare claims"], network=False)
    document = data.write_provenance(sources, tmp_path / "data_provenance.json")

    assert (tmp_path / "data_provenance.json").exists()
    assert document["all_inputs_real"] is False
    assert document["surrogate_count"] == 1


def test_one_source_string_splits_into_its_separate_inputs():
    parts = data.split_requirements(
        "Integrated panel dataset",
        "EPA AQS for PM2.5; CMS Medicare Claims for admissions; US Census ACS for deprivation",
    )
    assert len(parts) >= 3
