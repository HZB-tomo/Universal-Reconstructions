"""
Tests for PipelineRun._run_streaming() (ProjectConfig.stream=True):
reconstructs and writes one chunk of rows at a time instead of the
whole volume, driven from the same run_all() every caller (CLI,
run_batch(), the GUI's "Reconstruct selected") already goes through --
see orchestration/pipeline.py.

algotom's own fbp_reconstruction() is not bit-deterministic --
confirmed directly: calling this project's own eager reconstruct()
twice on the exact same normalized volume already gives a small but
real per-pixel difference. So "streaming produces the same
reconstruction as eager" can't mean bit-identical, or even "above a
fixed correlation value" -- how noisy a repeat run looks depends on
the phantom (a small, low-contrast one is noisier, relatively, than a
large one). _assert_matches_eager_noise_floor() below instead re-runs
the eager path a second time and requires streaming to correlate with
the first eager run at least as well as that second, independent eager
run does -- the real "same reconstruction" bar, self-calibrated per
test rather than a magic number that happens to work for one phantom
size. A real chunking/indexing bug would still show up clearly: it
produces a wrong or misaligned array, not just added noise, so its
correlation drops far below the self-noise floor, not just slightly.
"""
import json

import h5py
import numpy as np
import pytest

from orchestration.config import ProjectConfig
from orchestration.pipeline import PipelineRun
from tests.fixtures.realistic_phantom import generate as generate_phantom
from tests.fixtures.realistic_phantom import write_hdf5

algotom = pytest.importorskip("algotom", reason="algotom not installed -- these tests need the algotom_fbp backend")


def _write_phantom_hdf5(tmp_path, name="scan.h5", **kwargs):
    phantom = generate_phantom(**kwargs)
    path = tmp_path / name
    write_hdf5(phantom, path)
    return path, phantom


def _corr(a, b) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def _assert_matches_eager_noise_floor(eager_config: ProjectConfig, eager_volume, stream_volume, margin=0.03):
    repeat_result = PipelineRun(eager_config).run_all()
    assert repeat_result.success, repeat_result.error
    with h5py.File(repeat_result.volume_path) as f:
        eager_repeat_volume = f["/entry/volume/data"][:]

    baseline = _corr(eager_volume, eager_repeat_volume)
    cross = _corr(eager_volume, stream_volume)
    assert cross >= baseline - margin, (
        f"streaming correlates with eager ({cross:.6f}) noticeably worse than eager correlates with itself "
        f"on a repeat run ({baseline:.6f}) -- suggests a real bug, not just FBP's own noise"
    )


def test_streaming_matches_eager_reconstruction(tmp_path):
    path, _ = _write_phantom_hdf5(
        tmp_path, shape=(12, 48, 48), n_angles=150, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=1,
    )
    eager_config = ProjectConfig(
        path=str(path), format="hdf5", center=24.0, tilt_deg=0.0,
        export_format="hdf5", export_path=str(tmp_path / "eager.h5"),
    )
    eager_result = PipelineRun(eager_config).run_all()
    assert eager_result.success, eager_result.error

    stream_config = ProjectConfig(
        path=str(path), format="hdf5", storage_mode="disk", center=24.0, tilt_deg=0.0,
        stream=True, stream_chunk_rows=3,
        export_format="hdf5", export_path=str(tmp_path / "stream.h5"),
    )
    stream_result = PipelineRun(stream_config).run_all()
    assert stream_result.success, stream_result.error

    with h5py.File(eager_result.volume_path) as f:
        eager_volume = f["/entry/volume/data"][:]
    with h5py.File(stream_result.volume_path) as f:
        stream_volume = f["/entry/volume/data"][:]

    assert eager_volume.shape == stream_volume.shape == (12, 48, 48)
    _assert_matches_eager_noise_floor(eager_config, eager_volume, stream_volume)


def test_streaming_matches_eager_with_rows_crop_and_uint16(tmp_path):
    path, _ = _write_phantom_hdf5(
        tmp_path, shape=(16, 48, 48), n_angles=150, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=2,
    )
    common = dict(
        path=str(path), format="hdf5", storage_mode="disk", center=24.0, tilt_deg=0.0,
        rows=slice(2, 14), crop=((1, 9), (5, 40), (5, 40)),
        dtype="uint16", dtype_clip_min=-0.05, dtype_clip_max=0.3,
        export_format="hdf5",
    )
    eager_config = ProjectConfig(**common, export_path=str(tmp_path / "eager.h5"))
    eager_result = PipelineRun(eager_config).run_all()
    assert eager_result.success, eager_result.error

    stream_result = PipelineRun(
        ProjectConfig(**common, stream=True, stream_chunk_rows=2, export_path=str(tmp_path / "stream.h5")),
    ).run_all()
    assert stream_result.success, stream_result.error

    with h5py.File(eager_result.volume_path) as f:
        eager_volume = f["/entry/volume/data"][:]
        settings = json.loads(f["/entry/volume/data"].attrs["reconstruction_settings"])
    with h5py.File(stream_result.volume_path) as f:
        stream_volume = f["/entry/volume/data"][:]

    assert eager_volume.shape == stream_volume.shape == (8, 35, 35)
    assert eager_volume.dtype == stream_volume.dtype == np.uint16
    _assert_matches_eager_noise_floor(eager_config, eager_volume, stream_volume, margin=0.05)
    assert settings["crop"] == [[1, 9], [5, 40], [5, 40]]
    assert settings["dtype_clip"]["clip_min"] == -0.05


def test_streaming_matches_eager_with_nonzero_tilt(tmp_path):
    path, _ = _write_phantom_hdf5(
        tmp_path, shape=(16, 48, 48), n_angles=150, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=3,
    )
    common = dict(
        path=str(path), format="hdf5", storage_mode="disk", center=24.0, tilt_deg=0.8,
        rows=slice(0, 16), export_format="hdf5",
    )
    eager_config = ProjectConfig(**common, export_path=str(tmp_path / "eager.h5"))
    eager_result = PipelineRun(eager_config).run_all()
    assert eager_result.success, eager_result.error
    stream_result = PipelineRun(
        ProjectConfig(**common, stream=True, stream_chunk_rows=4, export_path=str(tmp_path / "stream.h5")),
    ).run_all()
    assert stream_result.success, stream_result.error

    with h5py.File(eager_result.volume_path) as f:
        eager_volume = f["/entry/volume/data"][:]
    with h5py.File(stream_result.volume_path) as f:
        stream_volume = f["/entry/volume/data"][:]
    _assert_matches_eager_noise_floor(eager_config, eager_volume, stream_volume)


def test_streaming_tiff_export_names_slices_by_absolute_row(tmp_path):
    path, _ = _write_phantom_hdf5(
        tmp_path, shape=(12, 32, 32), n_angles=100, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=4,
    )
    config = ProjectConfig(
        path=str(path), format="hdf5", storage_mode="disk", center=16.0, tilt_deg=0.0,
        rows=slice(3, 10), stream=True, stream_chunk_rows=2,
        export_format="tiff", export_path=str(tmp_path / "tiff_out"),
    )
    result = PipelineRun(config).run_all()
    assert result.success, result.error

    tiff_files = sorted((tmp_path / "tiff_out").glob("slice_*.tif"))
    assert len(tiff_files) == 7  # rows 3..9 inclusive
    assert tiff_files[0].name == "slice_00000.tif"
    assert tiff_files[-1].name == "slice_00006.tif"
    assert (tmp_path / "tiff_out" / "metadata.json").exists()


def test_streaming_requires_center_to_already_be_resolved(tmp_path):
    path, _ = _write_phantom_hdf5(
        tmp_path, shape=(6, 32, 32), n_angles=80, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=5,
    )
    config = ProjectConfig(
        path=str(path), format="hdf5", storage_mode="disk", center=None, tilt_deg=0.0,
        stream=True, export_path=str(tmp_path / "out.h5"),
    )
    result = PipelineRun(config).run_all()
    assert not result.success
    assert result.error.stage == "calibrate"
    assert "center" in str(result.error)


def test_streaming_uint16_requires_explicit_clip_bounds(tmp_path):
    path, _ = _write_phantom_hdf5(
        tmp_path, shape=(6, 32, 32), n_angles=80, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=6,
    )
    config = ProjectConfig(
        path=str(path), format="hdf5", storage_mode="disk", center=16.0, tilt_deg=0.0,
        stream=True, dtype="uint16", export_path=str(tmp_path / "out.h5"),
    )
    result = PipelineRun(config).run_all()
    assert not result.success
    assert result.error.stage == "postprocess"
    assert "dtype_clip" in str(result.error)


def test_streaming_defaults_missing_tilt_to_zero_instead_of_erroring(tmp_path):
    path, _ = _write_phantom_hdf5(
        tmp_path, shape=(6, 32, 32), n_angles=80, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=7,
    )
    config = ProjectConfig(
        path=str(path), format="hdf5", storage_mode="disk", center=16.0, tilt_deg=None,
        stream=True, export_path=str(tmp_path / "out.h5"),
    )
    result = PipelineRun(config).run_all()
    assert result.success, result.error
    with h5py.File(result.volume_path) as f:
        settings = json.loads(f["/entry/volume/data"].attrs["reconstruction_settings"])
    assert settings["tilt_deg"] == 0.0
