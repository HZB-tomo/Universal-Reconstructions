"""
Tests for orchestration/pipeline.py: PipelineRun's stage methods and
run_all(), against real files on disk (not in-memory ProjectObjects),
since that's what an actual orchestrated run looks like.
"""
import h5py
import numpy as np
import pytest

from orchestration.config import PipelineStageError, ProjectConfig
from orchestration.pipeline import PipelineRun
from tests.fixtures.generate_hypothetical_synchrotron import generate as generate_hypothetical
from tests.fixtures.realistic_phantom import generate as generate_phantom
from tests.fixtures.realistic_phantom import write_hdf5


def _write_phantom_hdf5(tmp_path, **kwargs):
    phantom = generate_phantom(**kwargs)
    path = tmp_path / "scan.h5"
    write_hdf5(phantom, path)
    return path, phantom


algotom = pytest.importorskip("algotom", reason="algotom not installed -- these tests need the algotom_fbp backend")


def test_run_all_succeeds_end_to_end_with_full_auto_calibration(tmp_path):
    path, phantom = _write_phantom_hdf5(
        tmp_path, shape=(8, 64, 64), n_angles=120, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=1,
    )
    config = ProjectConfig(
        path=str(path), format="hdf5",
        export_format="hdf5", export_path=str(tmp_path / "out.h5"),
    )

    result = PipelineRun(config).run_all()

    assert result.success, result.error
    assert result.volume_path == str(tmp_path / "out.h5")

    with h5py.File(result.volume_path, "r") as f:
        assert f["/volume"].shape == (8, 64, 64)
        assert f["/volume"].dtype == np.float32
        import json
        settings = json.loads(f["/volume"].attrs["reconstruction_settings"])
        assert settings["backend"] == "algotom_fbp"
        assert "center" in settings
        assert "tilt_deg" in settings


def test_run_all_honors_a_manually_supplied_center_and_tilt(tmp_path):
    path, _ = _write_phantom_hdf5(
        tmp_path, shape=(8, 64, 64), n_angles=120, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=1,
    )
    config = ProjectConfig(
        path=str(path), format="hdf5", center=41.5, tilt_deg=0.0,
        export_format="hdf5", export_path=str(tmp_path / "out.h5"),
    )

    result = PipelineRun(config).run_all()

    assert result.success, result.error
    with h5py.File(result.volume_path, "r") as f:
        import json
        settings = json.loads(f["/volume"].attrs["reconstruction_settings"])
        assert settings["center"] == 41.5


def test_run_all_with_rotate_deg_records_settings_and_uses_sinogram_reorder(tmp_path):
    # 120 angles over 180deg (the default span) -- enough coverage for
    # reconstruction.rotation.supports_exact_rotation() to be True, so
    # this should go through the sinogram-reorder path, not the
    # image-domain fallback.
    path, _ = _write_phantom_hdf5(
        tmp_path, shape=(8, 64, 64), n_angles=120, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=1,
    )
    config = ProjectConfig(
        path=str(path), format="hdf5", center=32.0, tilt_deg=0.0, rotate_deg=17.0,
        export_format="hdf5", export_path=str(tmp_path / "out.h5"),
    )

    run = PipelineRun(config)
    result = run.run_all()

    assert result.success, result.error
    assert any("sinogram reorder" in line for line in result.log)
    with h5py.File(result.volume_path, "r") as f:
        import json
        settings = json.loads(f["/volume"].attrs["reconstruction_settings"])
        assert settings["rotate_deg"] == 17.0


def test_run_all_with_rotate_deg_on_a_limited_angle_scan_falls_back(tmp_path):
    # 170deg span, short of the 180deg supports_exact_rotation() needs
    # -- must fall back to rotating the reconstructed volume, not error.
    path, _ = _write_phantom_hdf5(
        tmp_path, shape=(4, 48, 48), n_angles=80, angle_span=170.0, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=2,
    )
    config = ProjectConfig(
        path=str(path), format="hdf5", center=24.0, tilt_deg=0.0, rotate_deg=12.0,
        export_format="hdf5", export_path=str(tmp_path / "out.h5"),
    )

    result = PipelineRun(config).run_all()

    assert result.success, result.error
    assert any("image-domain fallback" in line for line in result.log)
    with h5py.File(result.volume_path, "r") as f:
        assert f["/volume"].shape == (4, 48, 48)


def test_streaming_with_rotate_deg_matches_eager_rotate_deg(tmp_path):
    # algotom_fbp isn't bit-deterministic call to call (see
    # test_streaming.py's module docstring), so this uses the same
    # self-calibrated noise-floor comparison: streaming-with-rotate_deg
    # must correlate with one eager-with-rotate_deg run at least as
    # well as a second, independent eager-with-rotate_deg run does.
    path, _ = _write_phantom_hdf5(
        tmp_path, shape=(12, 48, 48), n_angles=150, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=3,
    )
    common = dict(path=str(path), format="hdf5", center=24.0, tilt_deg=0.0, rotate_deg=9.0)

    def _corr(a, b):
        return float(np.corrcoef(a.ravel().astype(np.float64), b.ravel().astype(np.float64))[0, 1])

    eager_config = ProjectConfig(**common, export_format="hdf5", export_path=str(tmp_path / "eager.h5"))
    eager_result = PipelineRun(eager_config).run_all()
    assert eager_result.success, eager_result.error
    with h5py.File(eager_result.volume_path) as f:
        eager_volume = f["/volume"][:]

    eager_repeat_config = ProjectConfig(**common, export_format="hdf5", export_path=str(tmp_path / "eager_repeat.h5"))
    eager_repeat_result = PipelineRun(eager_repeat_config).run_all()
    assert eager_repeat_result.success, eager_repeat_result.error
    with h5py.File(eager_repeat_result.volume_path) as f:
        eager_repeat_volume = f["/volume"][:]

    stream_config = ProjectConfig(
        **common, stream=True, export_format="hdf5", export_path=str(tmp_path / "stream.h5"),
    )
    stream_result = PipelineRun(stream_config).run_all()
    assert stream_result.success, stream_result.error
    with h5py.File(stream_result.volume_path) as f:
        stream_volume = f["/volume"][:]

    baseline = _corr(eager_volume, eager_repeat_volume)
    cross = _corr(eager_volume, stream_volume)
    assert cross >= baseline - 0.03


def test_calibrate_writes_results_back_onto_the_project_object(tmp_path):
    path, _ = _write_phantom_hdf5(
        tmp_path, shape=(8, 64, 64), n_angles=120, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=1,
    )
    config = ProjectConfig(path=str(path), format="hdf5", center=41.5, tilt_deg=0.3)
    run = PipelineRun(config)
    run.import_data()
    run.complete()
    run.preprocess()

    run.calibrate()

    assert run.project.metadata.get("center_of_rotation") == 41.5
    assert run.project.metadata.get("tilt_angle") == 0.3
    assert run.project.metadata.get_field("center_of_rotation").source == "computed"


def test_postprocess_applies_uint16_conversion(tmp_path):
    config = ProjectConfig(path="unused.h5", dtype="uint16", dtype_clip_min=0.0, dtype_clip_max=1.0)
    run = PipelineRun(config)
    run.volume = np.array([[-1.0, 0.0], [0.5, 2.0]], dtype=np.float32)

    run.postprocess()

    assert run.volume.dtype == np.uint16
    assert run.settings["dtype_clip"] == {"clip_min": 0.0, "clip_max": 1.0, "source_dtype": "float32"}


def test_postprocess_applies_crop(tmp_path):
    config = ProjectConfig(path="unused.h5", crop=((1, 3), (0, 2), (0, 2)))
    run = PipelineRun(config)
    run.volume = np.arange(4 * 2 * 2, dtype=np.float32).reshape(4, 2, 2)

    run.postprocess()

    assert run.volume.shape == (2, 2, 2)
    assert run.settings["crop"] == ((1, 3), (0, 2), (0, 2))


def test_complete_raises_when_a_mandatory_field_cannot_be_resolved(tmp_path):
    path = tmp_path / "scan.h5"
    generate_hypothetical(path, shape=(4, 16, 16), n_angles=10)  # geometry_type never set
    config = ProjectConfig(path=str(path), format="hdf5")
    run = PipelineRun(config)
    run.import_data()

    with pytest.raises(PipelineStageError) as exc_info:
        run.complete()

    assert exc_info.value.stage == "complete"
    assert "geometry_type" in exc_info.value.fields


def test_complete_succeeds_when_config_answers_the_missing_field(tmp_path):
    path = tmp_path / "scan.h5"
    generate_hypothetical(path, shape=(4, 16, 16), n_angles=10)
    config = ProjectConfig(path=str(path), format="hdf5", completion_answers={"geometry_type": "parallel"})
    run = PipelineRun(config)
    run.import_data()

    run.complete()  # should not raise

    assert run.project.metadata.get("geometry_type") == "parallel"


def test_complete_prompts_when_interactive_and_config_lacks_the_answer(tmp_path, monkeypatch):
    path = tmp_path / "scan.h5"
    generate_hypothetical(path, shape=(4, 16, 16), n_angles=10)
    config = ProjectConfig(path=str(path), format="hdf5")
    run = PipelineRun(config, interactive=True)
    run.import_data()

    monkeypatch.setattr("builtins.input", lambda prompt: "parallel")
    run.complete()

    assert run.project.metadata.get("geometry_type") == "parallel"


def test_export_path_template_resolves_placeholders_from_settings(tmp_path):
    config = ProjectConfig(path="unused.h5", export_path=str(tmp_path / "{project_id}_{backend}.h5"))
    run = PipelineRun(config)
    run.project_id = "scanA"
    run.settings = {"backend": "algotom_fbp"}

    resolved = run._resolve_export_path(config.export_path)

    assert resolved == str(tmp_path / "scanA_algotom_fbp.h5")


def test_export_path_falls_back_to_nan_for_an_unresolved_placeholder(tmp_path):
    config = ProjectConfig(path="unused.h5", export_path=str(tmp_path / "{project_id}_{never_set}.h5"))
    run = PipelineRun(config)
    run.project_id = "scanA"
    run.settings = {}

    resolved = run._resolve_export_path(config.export_path)

    assert resolved == str(tmp_path / "scanA_NAN.h5")


def test_export_path_numbers_a_collision_instead_of_overwriting(tmp_path):
    config = ProjectConfig(path="unused.h5", export_path=str(tmp_path / "{project_id}.h5"))
    run = PipelineRun(config)
    run.project_id = "scanA"
    run.settings = {}
    (tmp_path / "scanA.h5").write_bytes(b"")

    resolved = run._resolve_export_path(config.export_path)

    assert resolved == str(tmp_path / "scanA_1.h5")
