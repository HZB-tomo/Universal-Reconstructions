"""
Tests for reconstruction/export.py: save_volume() writing a
reconstructed volume to disk as a TIFF stack or an HDF5 dataset, its
optional JSON settings/provenance record, and round-tripping cleanly
back through data_import's TiffBackend.
"""
import json

import h5py
import numpy as np
import pytest

from reconstruction import save_volume


def test_save_volume_writes_a_tiff_stack_readable_by_tifffile(tmp_path):
    tifffile = pytest.importorskip("tifffile")

    volume = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    folder = tmp_path / "out"

    save_volume(volume, folder, format="tiff")

    files = sorted(folder.glob("*.tif"))
    assert len(files) == 2
    for i, f in enumerate(files):
        np.testing.assert_array_equal(tifffile.imread(f), volume[i])


def test_save_volume_pads_a_single_2d_image_to_a_one_slice_stack(tmp_path):
    tifffile = pytest.importorskip("tifffile")

    image = np.arange(3 * 3, dtype=np.float32).reshape(3, 3)
    folder = tmp_path / "out"

    save_volume(image, folder, format="tiff")

    files = sorted(folder.glob("*.tif"))
    assert len(files) == 1
    np.testing.assert_array_equal(tifffile.imread(files[0]), image)


def test_save_volume_tiff_clears_stale_slices_from_a_previous_larger_export(tmp_path):
    pytest.importorskip("tifffile")

    folder = tmp_path / "out"
    save_volume(np.zeros((5, 2, 2), dtype=np.float32), folder, format="tiff")
    assert len(list(folder.glob("slice_*.tif"))) == 5

    save_volume(np.zeros((2, 2, 2), dtype=np.float32), folder, format="tiff")

    files = sorted(folder.glob("slice_*.tif"))
    assert len(files) == 2


def test_save_volume_tiff_output_round_trips_through_the_tiff_backend(tmp_path):
    pytest.importorskip("tifffile")
    from data_import.backend import TiffBackend

    volume = np.arange(3 * 5 * 5, dtype=np.float32).reshape(3, 5, 5)
    folder = tmp_path / "out"

    save_volume(volume, folder, format="tiff")

    handle = TiffBackend().build_data_handle(folder, "*.tif", storage_mode="ram")
    np.testing.assert_array_equal(handle.get(), volume)


def test_save_volume_writes_an_hdf5_dataset(tmp_path):
    volume = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    path = tmp_path / "out.h5"

    save_volume(volume, path, format="hdf5")

    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["/volume"][()], volume)


def test_save_volume_hdf5_honors_a_custom_dataset_path(tmp_path):
    volume = np.zeros((2, 3, 3), dtype=np.float32)
    path = tmp_path / "out.h5"

    save_volume(volume, path, format="hdf5", dataset_path="/entry/data")

    with h5py.File(path, "r") as f:
        assert "/entry/data" in f
        assert f["/entry/data"].shape == volume.shape


def test_save_volume_creates_missing_parent_directories(tmp_path):
    volume = np.zeros((1, 2, 2), dtype=np.float32)
    path = tmp_path / "a" / "b" / "out.h5"

    save_volume(volume, path, format="hdf5")

    assert path.exists()


def test_save_volume_rejects_an_unknown_format(tmp_path):
    volume = np.zeros((1, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="npy"):
        save_volume(volume, tmp_path / "out", format="npy")


def test_save_volume_writes_settings_as_an_hdf5_attribute(tmp_path):
    volume = np.zeros((1, 2, 2), dtype=np.float32)
    path = tmp_path / "out.h5"
    settings = {"backend": "algotom_fbp", "center": 56.0, "tilt_deg": 10.3, "ring_correction": True}

    save_volume(volume, path, format="hdf5", settings=settings)

    with h5py.File(path, "r") as f:
        stored = json.loads(f["/volume"].attrs["reconstruction_settings"])
        assert stored == settings


def test_save_volume_omits_the_hdf5_attribute_when_no_settings_given(tmp_path):
    volume = np.zeros((1, 2, 2), dtype=np.float32)
    path = tmp_path / "out.h5"

    save_volume(volume, path, format="hdf5")

    with h5py.File(path, "r") as f:
        assert "reconstruction_settings" not in f["/volume"].attrs


def test_save_volume_writes_settings_as_a_tiff_sidecar_json_file(tmp_path):
    pytest.importorskip("tifffile")

    volume = np.zeros((1, 2, 2), dtype=np.float32)
    folder = tmp_path / "out"
    settings = {"backend": "algotom_fbp", "crop": [[4, 8], [0, 96], [0, 96]]}

    save_volume(volume, folder, format="tiff", settings=settings)

    stored = json.loads((folder / "metadata.json").read_text())
    assert stored == settings


def test_save_volume_omits_the_tiff_sidecar_when_no_settings_given(tmp_path):
    pytest.importorskip("tifffile")

    volume = np.zeros((1, 2, 2), dtype=np.float32)
    folder = tmp_path / "out"

    save_volume(volume, folder, format="tiff")

    assert not (folder / "metadata.json").exists()


def test_save_volume_settings_accepts_numpy_scalars_and_arrays(tmp_path):
    volume = np.zeros((1, 2, 2), dtype=np.float32)
    path = tmp_path / "out.h5"
    settings = {"center": np.float64(56.0), "angles": np.array([0.0, 1.5, 3.0])}

    save_volume(volume, path, format="hdf5", settings=settings)

    with h5py.File(path, "r") as f:
        stored = json.loads(f["/volume"].attrs["reconstruction_settings"])
        assert stored == {"center": 56.0, "angles": [0.0, 1.5, 3.0]}
