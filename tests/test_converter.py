"""data_import/converter.py: plugin-read project -> one HDF5 file, with binning."""
import json

import h5py
import numpy as np
import pytest
import tifffile

from data_import.converter import ConversionCancelled, ConversionError, convert_to_hdf5
from data_import.importer import DataImporter
from tests.fixtures.realistic_phantom import generate, write_tiff

XML = '<scanInformation projectName="t"><imager pixelSize_mm="0.05"/></scanInformation>'


@pytest.fixture
def ill_folder(tmp_path):
    rng = np.random.default_rng(0)
    folder = tmp_path / "ill"
    (folder / "Proj").mkdir(parents=True)
    projs = rng.integers(0, 4000, (12, 20, 30), dtype=np.uint16)
    for i, p in enumerate(projs):
        tifffile.imwrite(folder / "Proj" / f"proj_{i:04d}.tif", p)
    flats = rng.integers(3000, 4000, (3, 20, 30), dtype=np.uint16)
    darks = rng.integers(0, 100, (2, 20, 30), dtype=np.uint16)
    tifffile.imwrite(folder / "FlatsNeutrons.tif", flats, photometric="minisblack")  # multi-page
    tifffile.imwrite(folder / "DarksNeutrons.tif", darks, photometric="minisblack")
    (folder / "scan_information.xml").write_text(XML)
    return folder, projs, flats, darks


def test_converts_an_ill_project_unbinned(ill_folder, tmp_path):
    folder, projs, flats, darks = ill_folder
    out = convert_to_hdf5(folder, tmp_path / "out.h5")
    with h5py.File(out, "r") as f:
        np.testing.assert_array_equal(f["/entry/instrument/detector/data"][()], projs)
        np.testing.assert_array_equal(f["/entry/instrument/detector/flat_fields"][()], flats)
        np.testing.assert_array_equal(f["/entry/instrument/detector/dark_fields"][()], darks)
        assert f["/entry/geometry/voxel_size"][()] == pytest.approx(50.0)
        assert "/entry/sample/rotation_angle" not in f  # the ILL plugin supplies no angles
        assert f["/entry/provenance"].attrs["source_plugin"] == "ill_neutrons"
        assert json.loads(f["/entry/provenance"].attrs["source_metadata"])["voxel_size"] == pytest.approx(50.0)


def test_binning_applies_to_projections_flats_and_darks_and_scales_pixel_size(ill_folder, tmp_path):
    folder, projs, *_ = ill_folder
    out = convert_to_hdf5(folder, tmp_path / "b.h5", bin_y=2, bin_x=2)
    with h5py.File(out, "r") as f:
        assert f["/entry/instrument/detector/data"].shape == (12, 10, 15)
        assert f["/entry/instrument/detector/flat_fields"].shape == (3, 10, 15)
        assert f["/entry/instrument/detector/dark_fields"].shape == (2, 10, 15)
        assert f["/entry/geometry/voxel_size"][()] == pytest.approx(100.0)
        assert f["/entry/instrument/detector/data"][5, 0, 0] == np.rint(projs[5, :2, :2].mean())
        assert tuple(f["/entry/provenance"].attrs["original_image_shape"]) == (20, 30)


def test_converted_file_imports_without_angles(ill_folder, tmp_path):
    folder, *_ = ill_folder
    out = convert_to_hdf5(folder, tmp_path / "i.h5", bin_y=2, bin_x=2)
    project = DataImporter().import_project(out, format="hdf5", storage_mode="ram")
    assert project.vendor_plugin_name == "urct_hdf5"
    assert "angles" in project.missing_soft_fields()
    project.close()


def test_converts_a_tiff_project_with_angles(tmp_path):
    phantom = generate(shape=(4, 16, 16), n_angles=20, noise_level=0.0, seed=1)
    folder = tmp_path / "phantom_tiff"
    write_tiff(phantom, folder)
    out = convert_to_hdf5(folder, tmp_path / "p.h5", bin_y=2, bin_x=2)
    with h5py.File(out, "r") as f:
        assert f["/entry/instrument/detector/data"].shape[0] == 20
        np.testing.assert_allclose(f["/entry/sample/rotation_angle"][()], phantom.angles)
    project = DataImporter().import_project(out, format="hdf5", storage_mode="ram")
    assert "angles" not in project.missing_soft_fields()
    assert "voxel_size" not in project.missing_soft_fields()
    project.close()


def test_refuses_to_overwrite_and_cleans_up_on_failure(ill_folder, tmp_path):
    folder, *_ = ill_folder
    out = tmp_path / "x.h5"
    convert_to_hdf5(folder, out)
    with pytest.raises(ConversionError, match="already exists"):
        convert_to_hdf5(folder, out)
    convert_to_hdf5(folder, out, overwrite=True)

    bad = tmp_path / "bad.h5"
    with pytest.raises(ConversionError):
        convert_to_hdf5(folder, bad, bin_y=64, bin_x=64)
    assert not bad.exists()


def test_cancel_removes_the_partial_file_and_progress_is_reported(ill_folder, tmp_path):
    folder, *_ = ill_folder
    seen = []
    out = tmp_path / "c.h5"
    with pytest.raises(ConversionCancelled):
        convert_to_hdf5(folder, out, batch=4, progress=lambda d, t, label: seen.append((d, t, label)),
                        cancel=lambda: len(seen) >= 1)
    assert not out.exists()
    assert seen[0] == (4, 17, "projections")


def test_tomo_convert_cli(ill_folder, tmp_path, capsys):
    from orchestration.cli import main

    folder, *_ = ill_folder
    out = tmp_path / "cli.h5"
    assert main(["convert", str(folder), str(out), "--bin", "2"]) == 0
    with h5py.File(out, "r") as f:
        assert f["/entry/instrument/detector/data"].shape == (12, 10, 15)
    assert main(["convert", str(folder), str(out)]) == 1  # exists, no --overwrite
    assert "already exists" in capsys.readouterr().err
    assert main(["convert", str(folder), str(out), "--overwrite", "--plugin", "ill_neutrons"]) == 0
