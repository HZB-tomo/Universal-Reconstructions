"""Phase retrieval inside the pipeline: config, metadata, eager vs streamed, row ranges, INI, previews."""
import h5py
import numpy as np
import pytest

from core.known_fields import KNOWN_FIELDS
from gui.preview_engine import PreviewEngine
from orchestration import PipelineRun, PipelineStageError, ProjectConfig, load_config, save_config
from orchestration.preprocess import preprocess_rows, resolve_phase
from tests.fixtures.realistic_phantom import generate as generate_phantom
from tests.fixtures.realistic_phantom import write_hdf5

pytest.importorskip("algotom")

PHASE = dict(
    phase_retrieval=True, phase_delta_beta=150.0, phase_distance_mm=100.0, phase_energy_kev=25.0, phase_pixel_um=2.0,
)


@pytest.fixture
def scan(tmp_path):
    path = tmp_path / "scan.h5"
    write_hdf5(generate_phantom(
        shape=(40, 48, 48), n_angles=60, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=1,
    ), path)
    return path


def _config(scan, **kwargs):
    base = dict(path=str(scan), format="hdf5", storage_mode="disk", center=24.0, tilt_deg=0.0)
    return ProjectConfig(**{**base, **kwargs})


def _prepared(scan, **kwargs):
    config = _config(scan, **kwargs)
    run = PipelineRun(config)
    run.import_data()
    run.complete()
    run.reference = None
    return run, config


def test_new_metadata_field_is_known_and_optional():
    field = KNOWN_FIELDS["propagation_distance"]
    assert field.tier == "optional" and field.unit == "mm" and "ask" in field.strategies


def test_config_defaults_leave_phase_retrieval_off():
    config = ProjectConfig()
    assert config.phase_retrieval is False and config.phase_method == "paganin"
    assert resolve_phase(None, config) is None


def test_missing_physics_is_reported_by_name_not_guessed(scan):
    run, config = _prepared(scan, phase_retrieval=True)
    with pytest.raises(PipelineStageError) as error:
        resolve_phase(run.project, config)
    message = str(error.value)
    for name in ("delta_over_beta", "distance_mm", "energy_kev"):
        assert name in message
    assert "pixel_um" not in message           # this file records its pixel size, so that one is known


def test_a_silent_default_pixel_size_does_not_count(scan):
    run, config = _prepared(scan, phase_retrieval=True, phase_delta_beta=100.0, phase_distance_mm=50.0, phase_energy_kev=20.0)
    run.project.metadata.set("voxel_size", 1.0, tier="mandatory_soft", source="default")   # as if the file had none
    with pytest.raises(PipelineStageError, match="pixel_um"):
        resolve_phase(run.project, config)


def test_values_come_from_metadata_when_the_config_does_not_set_them(scan):
    run, config = _prepared(
        scan, phase_retrieval=True, phase_delta_beta=100.0,
        completion_answers={"energy": "30", "propagation_distance": "80", "voxel_size": "1.5"},
    )
    setup = resolve_phase(run.project, config)         # also accepts numbers written as text in an INI
    assert (setup.params.energy_kev, setup.params.distance_mm, setup.params.pixel_um) == (30.0, 80.0, 1.5)
    config.phase_energy_kev = 20.0                      # an explicit config value wins
    assert resolve_phase(run.project, config).params.energy_kev == 20.0


def test_unknown_method_and_bad_numbers_are_stage_errors(scan):
    run, config = _prepared(scan, **{**PHASE, "phase_method": "nope"})
    with pytest.raises(PipelineStageError, match="Registered"):
        resolve_phase(run.project, config)
    run, config = _prepared(scan, **{**PHASE, "phase_delta_beta": -1.0})
    with pytest.raises(PipelineStageError, match="delta/beta"):
        resolve_phase(run.project, config)


def test_preprocess_smooths_across_rows_and_leaves_the_off_case_alone(scan):
    plain = PipelineRun(_config(scan))
    plain.import_data(); plain.complete(); baseline = plain.preprocess().copy()
    phased = PipelineRun(_config(scan, **PHASE))
    phased.import_data(); phased.complete(); filtered = phased.preprocess()
    assert filtered.shape == baseline.shape and filtered.dtype == np.float32
    assert not np.allclose(filtered, baseline)
    assert np.abs(np.diff(filtered, axis=1)).mean() < np.abs(np.diff(baseline, axis=1)).mean()   # low-pass
    assert any("phase retrieval" in line for line in phased.log)


def test_a_row_range_uses_real_neighbouring_rows_not_just_its_own(scan):
    """Rows 15-25 of a run restricted to those rows equal rows 15-25 of the full run."""
    full = PipelineRun(_config(scan, **PHASE))
    full.import_data(); full.complete(); everything = full.preprocess()
    part = PipelineRun(_config(scan, **PHASE, rows=slice(15, 25)))
    part.import_data(); part.complete(); subset = part.preprocess()
    assert subset.shape[1] == 10
    np.testing.assert_allclose(subset, everything[:, 15:25], atol=3e-3 * np.ptp(everything))


def test_streaming_matches_eager_with_phase_retrieval(scan, tmp_path):
    def run(name, **extra):
        config = _config(scan, **PHASE, export_path=str(tmp_path / name), **extra)
        result = PipelineRun(config).run_all()
        assert result.success, result.error
        with h5py.File(result.volume_path, "r") as f:
            return f["/entry/volume/data"][:]

    eager = run("eager.h5")
    streamed = run("stream.h5", stream=True, stream_chunk_rows=7)
    plain = run("plain.h5", **{})  # same settings: sanity that two runs agree
    assert eager.shape == streamed.shape == (40, 48, 48)
    np.testing.assert_allclose(streamed, eager, rtol=2e-3, atol=2e-3 * np.abs(eager).max())
    np.testing.assert_allclose(plain, eager, rtol=1e-5, atol=1e-7)

    without = PipelineRun(_config(scan, export_path=str(tmp_path / "off.h5"))).run_all()
    with h5py.File(without.volume_path, "r") as f:
        assert not np.allclose(f["/entry/volume/data"][:], eager)       # the retrieval really took part


def test_a_missing_value_fails_the_streamed_run_before_anything_is_written(scan, tmp_path):
    config = _config(scan, phase_retrieval=True, stream=True, export_path=str(tmp_path / "never.h5"))
    result = PipelineRun(config).run_all()
    assert not result.success and "phase retrieval" in str(result.error)
    assert not (tmp_path / "never.h5").exists()


def test_preprocess_rows_without_phase_is_the_plain_normalization(scan):
    from preprocessing import build_reference, normalize_rows

    run, config = _prepared(scan)
    reference = build_reference(run.project)
    np.testing.assert_array_equal(
        preprocess_rows(run.project, config, reference, slice(3, 9)), normalize_rows(run.project, slice(3, 9), reference),
    )


def test_ini_round_trip_of_the_phase_settings(tmp_path):
    config = ProjectConfig(path="a.h5", **PHASE, phase_method="paganin")
    ini = tmp_path / "c.ini"
    save_config(config, ini)
    text = ini.read_text()
    for key in ("phase_retrieval = true", "phase_delta_beta = 150.0", "phase_distance_mm = 100.0",
                "phase_energy_kev = 25.0", "phase_pixel_um = 2.0"):
        assert key in text
    loaded = load_config(ini)
    for name in ("phase_retrieval", "phase_method", "phase_delta_beta", "phase_distance_mm", "phase_energy_kev", "phase_pixel_um"):
        assert getattr(loaded, name) == getattr(config, name)


def test_the_preview_engine_includes_phase_retrieval(scan):
    run, config = _prepared(scan)
    engine = PreviewEngine()
    plain = engine.compute(run.project, config, 20, 24.0)
    phased_config = _config(scan, **PHASE)
    phased = engine.compute(run.project, phased_config, 20, 24.0)
    assert engine.reconstructions == 2                  # a different setting is a different cache entry
    assert not np.allclose(plain, phased)
    # the preview of a slice equals that slice of the full pipeline result
    phased_config.rows = slice(20, 21)
    full = PipelineRun(phased_config)
    full.project = run.project
    full.preprocess(); full.calibrate()
    np.testing.assert_allclose(phased, full.reconstruct()[0], rtol=2e-3, atol=2e-3 * np.abs(phased).max())
