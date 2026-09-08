"""
Tests for orchestration/batch.py: validate_config()'s advisory-only
dry-run, and run_batch()'s independence guarantee -- every project gets
a real attempt regardless of what validation found, and a genuinely
unexpected exception in one project never stops the rest.
"""
import pytest

from orchestration.batch import run_batch, validate_config
from orchestration.config import ProjectConfig
from tests.fixtures.generate_hypothetical_synchrotron import generate as generate_hypothetical
from tests.fixtures.realistic_phantom import generate as generate_phantom
from tests.fixtures.realistic_phantom import write_hdf5

algotom = pytest.importorskip("algotom", reason="algotom not installed -- these tests need the algotom_fbp backend")


def _write_phantom_hdf5(tmp_path, name="scan.h5", **kwargs):
    kwargs.setdefault("shape", (8, 64, 64))
    kwargs.setdefault("n_angles", 120)
    kwargs.setdefault("n_scratches", 0)
    kwargs.setdefault("n_defective_pixels", 0)
    kwargs.setdefault("inhomogeneity_strength", 0.0)
    kwargs.setdefault("noise_level", 0.0)
    kwargs.setdefault("seed", 1)
    phantom = generate_phantom(**kwargs)
    path = tmp_path / name
    write_hdf5(phantom, path)
    return path


def test_validate_config_returns_empty_when_everything_resolves(tmp_path):
    path = _write_phantom_hdf5(tmp_path)
    config = ProjectConfig(path=str(path), format="hdf5")

    assert validate_config(config) == []


def test_validate_config_reports_a_missing_completion_field(tmp_path):
    path = tmp_path / "scan.h5"
    generate_hypothetical(path, shape=(4, 16, 16), n_angles=10)
    config = ProjectConfig(path=str(path), format="hdf5")

    problems = validate_config(config)

    assert problems
    assert "geometry_type" in problems[0]


def test_run_batch_runs_every_project_regardless_of_validation_problems(tmp_path):
    good_path = _write_phantom_hdf5(tmp_path, name="good.h5")
    bad_path = tmp_path / "bad.h5"
    generate_hypothetical(bad_path, shape=(4, 16, 16), n_angles=10)  # missing geometry_type

    good_config = ProjectConfig(
        path=str(good_path), format="hdf5", project_id="good",
        export_format="hdf5", export_path=str(tmp_path / "good_out.h5"),
    )
    bad_config = ProjectConfig(path=str(bad_path), format="hdf5", project_id="bad")

    report = run_batch([good_config, bad_config])

    # validation is advisory: both were attempted, not just the good one
    assert {r.project_id for r in report.results} == {"good", "bad"}
    assert "bad" in report.validation_problems
    assert "good" not in report.validation_problems

    assert [r.project_id for r in report.succeeded] == ["good"]
    assert [r.project_id for r in report.failed] == ["bad"]
    assert report.failed[0].error.stage == "complete"


def test_run_batch_survives_an_exception_pipelinestageerror_never_wraps(tmp_path):
    good_path = _write_phantom_hdf5(tmp_path, name="good.h5")
    missing_path = tmp_path / "does_not_exist.h5"

    good_config = ProjectConfig(
        path=str(good_path), format="hdf5", project_id="good",
        export_format="hdf5", export_path=str(tmp_path / "good_out.h5"),
    )
    # A genuinely nonexistent file: h5py raises OSError/FileNotFoundError
    # when opening it, which import_data() does NOT catch and wrap into
    # PipelineStageError -- exactly the "unanticipated failure" case
    # run_batch() itself has to guard against.
    missing_config = ProjectConfig(path=str(missing_path), format="hdf5", project_id="missing")

    report = run_batch([good_config, missing_config])

    assert {r.project_id for r in report.results} == {"good", "missing"}
    assert [r.project_id for r in report.succeeded] == ["good"]
    missing_result = next(r for r in report.results if r.project_id == "missing")
    assert not missing_result.success
    assert missing_result.error is not None


def test_pipeline_result_carries_its_config_for_retry(tmp_path):
    bad_path = tmp_path / "bad.h5"
    generate_hypothetical(bad_path, shape=(4, 16, 16), n_angles=10)
    bad_config = ProjectConfig(path=str(bad_path), format="hdf5", project_id="bad")

    report = run_batch([bad_config])

    assert report.failed[0].config is bad_config

    # the natural retry pattern, now that the config is fixed
    fixed = report.failed[0].config
    fixed.completion_answers = {"geometry_type": "parallel"}
    assert validate_config(fixed) == []
