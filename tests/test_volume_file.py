"""The volume file of the URCT HDF5 standard: chunks, record, measurement, streaming writer, reader."""
import json

import h5py
import numpy as np
import pytest

from core import urct_format as fmt
from reconstruction import StreamingVolumeWriter, save_volume
from reconstruction.volume_file import chunk_shape, find_volume_dataset, read_volume_info

SETTINGS = {
    "backend": "algotom_fbp", "filter_name": "hamming", "num_iter": 20, "center": 31.5, "tilt_deg": 0.25,
    "rotate_deg": 12.0, "crop": ((0, 4), (0, 8), (0, 8)), "device_used": "cpu", "speed_slices_per_s": 3.5,
    "preprocessing": {"ring_correction": True, "ring_method": "sorting", "phase_retrieval": {"delta_over_beta": 100.0}},
    "geometry_type": "parallel", "angles": np.linspace(0, 180, 90, endpoint=False), "voxel_size": 2.5,
    "detector_pixel_size": 2.5, "energy": 18.0, "sample_name": "rock",
    "source_file_uuid": "1234", "source_file": "scan.h5",
}


def test_chunks_are_cubic_and_clipped():
    assert chunk_shape((500, 500, 500), np.float32) == (64, 64, 64)
    assert chunk_shape((500, 500, 500), np.uint16) == (64, 64, 128)
    assert chunk_shape((10, 300, 20), np.float32) == (10, 64, 20)


def test_a_saved_volume_is_a_standard_file(tmp_path):
    volume = np.random.default_rng(0).random((70, 80, 90), dtype=np.float32)
    path = tmp_path / "v.h5"
    save_volume(volume, path, "hdf5", settings=SETTINGS)
    with h5py.File(path, "r") as f:
        assert f.attrs["urct_file_type"] == "volume" and f.attrs["urct_standard"] == "urct-hdf5"
        data = f[fmt.VOLUME]
        assert data.chunks == (64, 64, 64) and data.compression is None
        np.testing.assert_array_equal(data[()], volume)
        assert data.attrs["axes"] == "z,y,x" and data.attrs["units"] == "1/cm" and data.attrs["voxel_size"] == 2.5
        assert f[fmt.SOFTWARE].attrs["role"] == "reconstructed_by" and "numpy" in f[fmt.SOFTWARE].attrs
        assert f["/entry/source"].attrs["uuid"] == "1234"
        assert json.loads(data.attrs["reconstruction_settings"])["backend"] == "algotom_fbp"   # nothing a caller passes is lost


def test_record_and_measurement_are_filed_at_the_standard_paths(tmp_path):
    path = tmp_path / "v.h5"
    save_volume(np.zeros((4, 8, 8), np.float32), path, "hdf5", settings=SETTINGS)
    with h5py.File(path, "r") as f:
        record = f["/entry/reconstruction"]
        assert record.attrs["algorithm"] == "algotom_fbp" and record.attrs["filter"] == "hamming"
        assert record.attrs["num_iter"] == 20 and record.attrs["center_of_rotation"] == 31.5
        assert record.attrs["tilt_angle"] == 0.25 and record.attrs["rotation_offset"] == 12.0
        assert record.attrs["device_used"] == "cpu" and record.attrs["speed_slices_per_s"] == 3.5
        assert record.attrs["crop"].shape == (3, 2)
        assert record["preprocessing"].attrs["ring_method"] == "sorting"
        assert record["preprocessing/phase_retrieval"].attrs["delta_over_beta"] == 100.0
        # the measurement, at the same paths as in a projection file
        assert f["/entry/geometry/geometry_type"].asstr()[()] == "parallel"
        assert f["/entry/geometry/voxel_size"][()] == 2.5 and f["/entry/instrument/source/energy"][()] == 18.0
        assert f["/entry/sample/rotation_angle"].shape == (90,) and f["/entry/sample/name"].asstr()[()] == "rock"
        assert f.attrs["urct_complete"]
    assert fmt.check_file(path).ok and fmt.check_file(path).file_type == "volume"


def test_a_per_slice_center_is_stored_as_a_dataset(tmp_path):
    path = tmp_path / "v.h5"
    centers = np.linspace(30, 33, 200)
    save_volume(np.zeros((2, 4, 4), np.float32), path, "hdf5", settings={"center": centers})
    with h5py.File(path, "r") as f:
        np.testing.assert_allclose(f["/entry/reconstruction/center_of_rotation"][()], centers)


def test_uint16_volumes_say_how_to_get_the_values_back(tmp_path):
    from reconstruction import to_uint16

    physical = np.random.default_rng(1).random((5, 6, 7), dtype=np.float32) * 3 - 1
    stored, clip = to_uint16(physical)
    path = tmp_path / "u.h5"
    save_volume(stored, path, "hdf5", settings={"dtype_clip": clip, "detector_pixel_size": 1.0})
    info = read_volume_info(path)
    assert info.dtype == "uint16" and info.units == "1/cm" and info.chunks == (5, 6, 7)
    assert info.record["clip_min"] == pytest.approx(clip["clip_min"])
    with h5py.File(path, "r") as f:
        back = info.to_physical(f[fmt.VOLUME][()])
    np.testing.assert_allclose(back, physical, atol=(clip["clip_max"] - clip["clip_min"]) / 65535)


def test_read_volume_info_round_trip_and_legacy_files(tmp_path):
    path = tmp_path / "v.h5"
    save_volume(np.zeros((3, 4, 5), np.float32), path, "hdf5", settings=SETTINGS)
    info = read_volume_info(path)
    assert info.shape == (3, 4, 5) and info.dataset_path == fmt.VOLUME and info.voxel_size == 2.5
    assert info.record["algorithm"] == "algotom_fbp" and info.record["preprocessing"]["ring_method"] == "sorting"
    assert info.measurement["geometry_type"] == "parallel" and info.measurement["sample_name"] == "rock"
    assert info.source["uuid"] == "1234" and info.software["urct"] and info.version == "0.1" and info.complete

    old = tmp_path / "old.h5"       # a file written before the standard: one /volume dataset, settings as JSON
    with h5py.File(old, "w") as f:
        f.create_dataset("/volume", data=np.zeros((2, 3, 4), np.float32)).attrs["reconstruction_settings"] = json.dumps({"backend": "x"})
        assert find_volume_dataset(f) == "/volume"
    legacy = read_volume_info(old)
    assert legacy.shape == (2, 3, 4) and legacy.record == {"backend": "x"} and legacy.measurement == {}

    empty = tmp_path / "e.h5"
    with h5py.File(empty, "w") as f:
        f.create_dataset("/other", data=1)
    with pytest.raises(ValueError, match="no volume dataset"):
        read_volume_info(empty)


@pytest.mark.parametrize("order", ["increasing", "reversed", "ragged"])
def test_streaming_writer_matches_save_volume_whatever_the_chunking(tmp_path, order):
    volume = np.random.default_rng(2).random((150, 12, 10), dtype=np.float32)
    path = tmp_path / "s.h5"
    sizes = {"increasing": [16] * 10, "reversed": [16] * 10, "ragged": [5, 70, 1, 30, 44]}[order]
    starts = np.concatenate([[0], np.cumsum(sizes)])[:-1].astype(int)
    pieces = [(int(s), min(int(s) + n, 150)) for s, n in zip(starts, sizes) if s < 150]
    if order == "reversed":
        pieces = pieces[::-1]
    with StreamingVolumeWriter(path, "hdf5", 150, 12, 10, np.float32, settings=SETTINGS) as writer:
        for a, b in pieces:
            writer.write_chunk(volume[a:b], z_offset=a)
        writer.update_record(duration_s=1.5, finished="today")
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f[fmt.VOLUME][()], volume)
        assert f[fmt.VOLUME].chunks == (64, 12, 10)
        assert f["/entry/reconstruction"].attrs["duration_s"] == 1.5 and f["/entry/reconstruction"].attrs["algorithm"] == "algotom_fbp"


def test_blocks_left_unfinished_by_a_stopped_run_are_still_written(tmp_path):
    volume = np.ones((100, 4, 4), np.float32)
    path = tmp_path / "p.h5"
    with StreamingVolumeWriter(path, "hdf5", 100, 4, 4, np.float32) as writer:
        writer.write_chunk(volume[:10], z_offset=0)
    with h5py.File(path, "r") as f:
        assert f[fmt.VOLUME][:10].min() == 1.0 and f[fmt.VOLUME][10:].max() == 0.0


def test_tiff_output_is_unchanged(tmp_path):
    save_volume(np.zeros((3, 4, 4), np.float32), tmp_path / "t", "tiff", settings={"backend": "x"})
    assert (tmp_path / "t" / "slice_00000.tif").exists()
    assert json.loads((tmp_path / "t" / "metadata.json").read_text())["backend"] == "x"


# -- through the pipeline ------------------------------------------------------

algotom = pytest.importorskip("algotom")


@pytest.mark.parametrize("stream", [False, True])
def test_a_pipeline_run_writes_a_traceable_volume(tmp_path, stream):
    from data_import.converter import convert_to_hdf5
    from orchestration.config import ProjectConfig
    from orchestration.pipeline import PipelineRun
    from tests.fixtures.realistic_phantom import generate, write_tiff

    phantom = generate(shape=(8, 48, 48), n_angles=60, cor_deviation=0.0, tilt_deg=0.0, n_scratches=0,
                       n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=3)
    write_tiff(phantom, tmp_path / "tiff")
    source = convert_to_hdf5(tmp_path / "tiff", tmp_path / "scan.h5")
    config = ProjectConfig(path=str(source), format="hdf5", export_path=str(tmp_path / "out.h5"),
                           center=23.5, tilt_deg=0.0, stream=stream, reconstruction_filter="hamming",
                           ring_correction=True, reconstruction_backend="algotom_fbp")
    result = PipelineRun(config).run_all()
    assert result.success, result.error

    info = read_volume_info(result.volume_path)
    assert info.shape == (8, 48, 48) and info.chunks == (8, 48, 48)
    assert info.record["algorithm"] == "algotom_fbp" and info.record["filter"] == "hamming"
    assert info.record["device_requested"] == "auto" and info.record["device_used"] in ("cpu", "gpu")
    assert info.record["center_of_rotation"] == 23.5
    assert info.record["speed_slices_per_s"] > 0 and info.record["n_slices"] == 8 and "started" in info.record
    assert info.record["preprocessing"]["ring_correction"] and info.record["preprocessing"]["ring_method"] == "sorting"
    assert info.measurement["geometry_type"] == "parallel"
    with h5py.File(source, "r") as f:
        assert info.source["uuid"] == f.attrs["urct_uuid"]                   # the link back to the projection file
    assert info.source["path"] == str(source)
