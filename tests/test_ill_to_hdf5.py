"""ill_to_hdf5: binning maths, ILL folder -> HDF5 conversion, and import of the result."""
import h5py
import numpy as np
import pytest
import tifffile

import ill_to_hdf5 as conv
from data_import.importer import DataImporter

XML = ('<scanInformation projectName="t"><imager pixelSize_mm="0.05"/>'
       '<acquisition numberOfProjections="12"/></scanInformation>')


@pytest.fixture
def ill_folder(tmp_path):
    rng = np.random.default_rng(0)
    (tmp_path / "Proj").mkdir()
    projs = rng.integers(0, 4000, (12, 20, 30), dtype=np.uint16)
    for i, p in enumerate(projs):
        tifffile.imwrite(tmp_path / "Proj" / f"proj_{i}.tif", p)  # NOT zero-padded on purpose
    flats = rng.integers(3000, 4000, (3, 20, 30), dtype=np.uint16)
    darks = rng.integers(0, 100, (2, 20, 30), dtype=np.uint16)
    tifffile.imwrite(tmp_path / "FlatsNeutrons.tif", flats, photometric="minisblack")
    tifffile.imwrite(tmp_path / "DarksNeutrons.tif", darks, photometric="minisblack")
    (tmp_path / "scan_information.xml").write_text(XML)
    return tmp_path, projs, flats, darks


def test_bin_frames_mean_and_sum():
    a = np.arange(2 * 4 * 6, dtype=np.float32).reshape(2, 4, 6)
    m = conv.bin_frames(a, 2, 3, "mean")
    assert m.shape == (2, 2, 2)
    np.testing.assert_allclose(m[0, 0, 0], a[0, :2, :3].mean())
    np.testing.assert_allclose(conv.bin_frames(a, 2, 3, "sum")[1, 1, 1], a[1, 2:, 3:].sum())
    assert conv.bin_frames(a[0], 2, 2).shape == (2, 3)


def test_bin_frames_centred_crop():
    a = np.zeros((1, 7, 7), np.float32)
    a[0, 3, 3] = 9.0  # centre pixel
    out = conv.bin_frames(a, 3, 3, "sum")  # 7 -> crop 1 total (0 front, 1 back) -> 2x2
    assert out.shape == (1, 2, 2)
    with pytest.raises(ValueError):
        conv.bin_frames(a, 8, 8)


def test_convert_unbinned_roundtrip_and_natural_order(ill_folder, tmp_path):
    folder, projs, flats, darks = ill_folder
    out = tmp_path / "out.h5"
    conv.convert(folder, out, angle_span=180.0, quiet=True)
    with h5py.File(out, "r") as f:
        np.testing.assert_array_equal(f[conv.PROJECTIONS][()], projs)  # proj_2 before proj_10
        np.testing.assert_array_equal(f[conv.FLAT_FIELDS][()], flats)
        np.testing.assert_array_equal(f[conv.DARK_FIELDS][()], darks)
        np.testing.assert_allclose(f[conv.ANGLES][()], np.arange(12) * 15.0)
        assert f[conv.PIXEL_SIZE][()] == pytest.approx(50.0)


def test_convert_binned_all_three_and_pixel_size(ill_folder, tmp_path):
    folder, projs, flats, darks = ill_folder
    out = tmp_path / "b.h5"
    conv.convert(folder, out, bin_y=2, bin_x=2, angle_span=180.0, quiet=True)
    with h5py.File(out, "r") as f:
        assert f[conv.PROJECTIONS].shape == (12, 10, 15)
        assert f[conv.FLAT_FIELDS].shape == (3, 10, 15)
        assert f[conv.DARK_FIELDS].shape == (2, 10, 15)
        assert f[conv.PIXEL_SIZE][()] == pytest.approx(100.0)
        expected = np.rint(projs[4, :2, :2].mean())
        assert f[conv.PROJECTIONS][4, 0, 0] == expected


def test_converted_file_imports(ill_folder, tmp_path):
    folder, *_ = ill_folder
    out = tmp_path / "i.h5"
    conv.convert(folder, out, bin_y=2, bin_x=2, angle_span=180.0, quiet=True)
    project = DataImporter().import_project(out, format="hdf5", storage_mode="ram")
    assert project.vendor_plugin_name == "generic_parallel_synchrotron_hdf5"
    assert project.is_ready_for_import()
    assert project.data_handles["projections"].shape[0] == 12
    project.close()


def test_failure_leaves_no_file(ill_folder, tmp_path):
    folder, *_ = ill_folder
    out = tmp_path / "bad.h5"
    with pytest.raises(ValueError):
        conv.convert(folder, out, bin_y=64, bin_x=64, angle_span=180.0, quiet=True)
    assert not out.exists()


def test_no_angles_by_default_and_still_imports(ill_folder, tmp_path):
    folder, *_ = ill_folder
    out = tmp_path / "noang.h5"
    conv.convert(folder, out, bin_y=2, bin_x=2, quiet=True)
    with h5py.File(out, "r") as f:
        assert conv.ANGLES not in f
        assert f.attrs["angles_source"] == "none"
    project = DataImporter().import_project(out, format="hdf5", storage_mode="ram")
    assert project.vendor_plugin_name == "generic_parallel_synchrotron_hdf5"
    assert "angles" in project.missing_soft_fields()
    project.close()
    with pytest.raises(ValueError, match="not both"):
        conv.convert(folder, tmp_path / "x.h5", angle_span=180.0, angles_file="a.txt", quiet=True)
