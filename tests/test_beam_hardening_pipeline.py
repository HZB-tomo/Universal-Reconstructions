"""Beam hardening correction inside the pipeline: config, validation, ordering, streaming, INI, previews."""
import h5py
import numpy as np
import pytest

from gui.preview_engine import PreviewEngine
from orchestration import PipelineRun, PipelineStageError, ProjectConfig, load_config, save_config
from orchestration.preprocess import preprocess_rows, resolve_beam_hardening
from preprocessing import build_reference, correct_beam_hardening
from tests.fixtures.realistic_phantom import generate as generate_phantom
from tests.fixtures.realistic_phantom import write_hdf5

pytest.importorskip("algotom")

COEFFICIENTS = (1.0, 0.3, -0.02)


@pytest.fixture
def scan(tmp_path):
    path = tmp_path / "scan.h5"
    write_hdf5(generate_phantom(
        shape=(20, 40, 40), n_angles=50, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=1,
    ), path)
    return path


def _config(scan, **kwargs):
    base = dict(path=str(scan), format="hdf5", storage_mode="disk", center=20.0, tilt_deg=0.0,
                reconstruction_backend="algotom_fbp")        # streamed and eager must agree exactly: a deterministic reference
    return ProjectConfig(**{**base, **kwargs})


def _prepared(scan, **kwargs):
    config = _config(scan, **kwargs)
    run = PipelineRun(config)
    run.import_data()
    run.complete()
    return run, config


def test_off_by_default():
    config = ProjectConfig()
    assert config.beam_hardening is False and config.bh_coefficients is None
    assert resolve_beam_hardening(config) is None


def test_missing_or_unusable_coefficients_are_stage_errors():
    with pytest.raises(PipelineStageError, match="no coefficients"):
        resolve_beam_hardening(ProjectConfig(beam_hardening=True))
    with pytest.raises(PipelineStageError, match="stop increasing"):
        resolve_beam_hardening(ProjectConfig(beam_hardening=True, bh_coefficients=(1.0, -0.5)))
    setup = resolve_beam_hardening(ProjectConfig(beam_hardening=True, bh_coefficients=(1, 0.3, -0.02)))
    assert setup.coefficients == (1.0, 0.3, -0.02)


def test_the_step_is_applied_last_in_preprocessing(scan):
    run, config = _prepared(scan)
    reference = build_reference(run.project)
    plain = preprocess_rows(run.project, config, reference, slice(5, 12))
    config.beam_hardening, config.bh_coefficients = True, COEFFICIENTS
    corrected = preprocess_rows(run.project, config, reference, slice(5, 12))
    assert not np.allclose(plain, corrected)
    np.testing.assert_array_equal(corrected, correct_beam_hardening(plain, COEFFICIENTS))


def test_it_follows_ring_removal_and_phase_retrieval(scan):
    """Order: normalize -> zinger -> phase -> ring -> beam hardening."""
    run, config = _prepared(
        scan, ring_correction=True, phase_retrieval=True, phase_delta_beta=100.0, phase_distance_mm=50.0,
        phase_energy_kev=25.0, phase_pixel_um=2.0,
    )
    reference = build_reference(run.project)
    before = preprocess_rows(run.project, config, reference, slice(5, 12))
    config.beam_hardening, config.bh_coefficients = True, COEFFICIENTS
    after = preprocess_rows(run.project, config, reference, slice(5, 12))
    np.testing.assert_array_equal(after, correct_beam_hardening(before, COEFFICIENTS))


def test_streaming_matches_eager_exactly_and_the_correction_takes_part(scan, tmp_path):
    def run(name, **extra):
        config = _config(scan, beam_hardening=True, bh_coefficients=COEFFICIENTS, export_path=str(tmp_path / name), **extra)
        result = PipelineRun(config).run_all()
        assert result.success, result.error
        with h5py.File(result.volume_path, "r") as f:
            return f["/entry/volume/data"][:]

    eager = run("eager.h5")
    streamed = run("stream.h5", stream=True, stream_chunk_rows=7)
    np.testing.assert_allclose(streamed, eager, rtol=1e-4, atol=1e-6 * np.abs(eager).max())   # a per-value operation
    off = PipelineRun(_config(scan, export_path=str(tmp_path / "off.h5"))).run_all()
    with h5py.File(off.volume_path, "r") as f:
        assert not np.allclose(f["/entry/volume/data"][:], eager)


def test_a_bad_setting_fails_the_streamed_run_before_anything_is_written(scan, tmp_path):
    config = _config(scan, beam_hardening=True, bh_coefficients=(1.0, -0.5), stream=True,
                     export_path=str(tmp_path / "never.h5"))
    result = PipelineRun(config).run_all()
    assert not result.success and "beam hardening" in str(result.error)
    assert not (tmp_path / "never.h5").exists()


def test_max_attenuation_option_is_honoured():
    config = ProjectConfig(beam_hardening=True, bh_coefficients=(1.0, 0.0, -0.02), bh_options={"max_attenuation": 2.0})
    assert resolve_beam_hardening(config).max_attenuation == 2.0       # a limit below the minimum is not an error


def test_ini_round_trip(tmp_path):
    config = ProjectConfig(path="a.h5", beam_hardening=True, bh_coefficients=(1.0, 0.35, -0.025))
    ini = tmp_path / "c.ini"
    save_config(config, ini)
    text = ini.read_text()
    assert "beam_hardening = true" in text and "bh_coefficients = 1.0, 0.35, -0.025" in text
    loaded = load_config(ini)
    assert loaded.beam_hardening is True and loaded.bh_coefficients == (1.0, 0.35, -0.025)

    bad = tmp_path / "bad.ini"
    bad.write_text("[preprocessing]\nbh_coefficients = 1, two\n")
    with pytest.raises(ValueError, match="comma-separated numbers"):
        load_config(bad)


def test_the_preview_engine_includes_the_correction(scan):
    run, config = _prepared(scan)
    engine = PreviewEngine()
    plain = engine.compute(run.project, config, 10, 20.0)
    corrected_config = _config(scan, beam_hardening=True, bh_coefficients=COEFFICIENTS)
    corrected = engine.compute(run.project, corrected_config, 10, 20.0)
    assert engine.reconstructions == 2                      # a different setting is a different cache entry
    assert not np.allclose(plain, corrected)
    corrected_config.rows = slice(10, 11)
    full = PipelineRun(corrected_config)
    full.project = run.project
    full.preprocess(); full.calibrate()
    np.testing.assert_allclose(corrected, full.reconstruct()[0], rtol=2e-3, atol=2e-3 * np.abs(corrected).max())
