"""The URCT HDF5 file standard: converter output, the urct_hdf5 plugin, validation, versioning."""
import re
from pathlib import Path

import h5py
import numpy as np
import pytest
import tifffile

from core import urct_format as fmt
from core.version import __version__
from data_import.converter import convert_to_hdf5
from data_import.importer import DataImporter
from tests.fixtures.realistic_phantom import generate, write_hdf5, write_tiff
from tests.test_converter import XML


@pytest.fixture
def converted(tmp_path):
    phantom = generate(shape=(4, 16, 16), n_angles=20, noise_level=0.0, seed=1)
    write_tiff(phantom, tmp_path / "t")
    return convert_to_hdf5(tmp_path / "t", tmp_path / "o.h5"), phantom


def write_minimal(path, **fields):
    """A standard file written by hand: projections plus the given fields (key -> value)."""
    with h5py.File(path, "w") as f:
        fmt.write_header(f, fmt.FILE_PROJECTIONS)
        f.create_dataset(fmt.PROJECTIONS, data=np.ones((6, 4, 5), dtype=np.float32))
        for key, value in fields.items():
            fmt.write_field(f, key, value)
    return path


def test_converter_writes_the_standard(converted):
    path, phantom = converted
    with h5py.File(path, "r") as f:
        assert f.attrs["urct_standard"] == "urct-hdf5" and f.attrs["urct_file_type"] == "projections"
        assert f.attrs["urct_version"] == "0.1" and len(f.attrs["urct_uuid"]) == 36
        assert f.attrs["urct_complete"]
        data = f[fmt.PROJECTIONS]
        assert data.compression is None and data.chunks == (1, 4, 16)         # no compression, one frame per chunk
        assert data.attrs["axes"] == "angle,row,column"
        assert f["/entry/geometry/geometry_type"].asstr()[()] == "parallel"
        np.testing.assert_allclose(f["/entry/sample/rotation_angle"][()], phantom.angles)
        assert f["/entry/sample/rotation_angle"].attrs["units"] == "degrees"
        assert f[fmt.SOFTWARE].attrs["urct"] == __version__ and f[fmt.SOFTWARE].attrs["role"] == "converted_by"
        assert "source_plugin" in f[fmt.PROVENANCE].attrs
    report = fmt.check_file(path)
    assert report.ok and report.complete and report.missing == [] and report.file_type == "projections"


def test_converter_leaves_out_what_the_source_does_not_know(tmp_path):
    rng = np.random.default_rng(0)
    folder = tmp_path / "ill"
    (folder / "Proj").mkdir(parents=True)
    for i in range(6):
        tifffile.imwrite(folder / "Proj" / f"proj_{i:04d}.tif", rng.integers(0, 4000, (8, 8), dtype=np.uint16))
    tifffile.imwrite(folder / "FlatsNeutrons.tif", np.full((2, 8, 8), 3000, np.uint16), photometric="minisblack")
    tifffile.imwrite(folder / "DarksNeutrons.tif", np.zeros((2, 8, 8), np.uint16), photometric="minisblack")
    (folder / "scan_information.xml").write_text(XML)
    out = convert_to_hdf5(folder, tmp_path / "i.h5")
    with h5py.File(out, "r") as f:
        assert "/entry/sample/rotation_angle" not in f and "/entry/geometry/angle_span" not in f
        assert not f.attrs["urct_complete"]
    report = fmt.check_file(out)
    assert report.ok and not report.complete and "angles or angle_span" in report.missing


def test_plugin_reads_everything_back(converted):
    path, phantom = converted
    project = DataImporter().import_project(path, format="hdf5", storage_mode="ram")
    assert project.vendor_plugin_name == "urct_hdf5"
    assert project.metadata.get("geometry_type") == "parallel"
    np.testing.assert_allclose(project.metadata.get("angles"), phantom.angles)
    assert project.metadata.get("voxel_size") is not None
    project.close()


def test_plugin_accepts_the_old_angle_path_and_a_lone_pixel_size(tmp_path):
    path = tmp_path / "old.h5"
    write_minimal(path, geometry_type="parallel", detector_pixel_size=7.5)
    with h5py.File(path, "a") as f:
        f.create_dataset("/entry/instrument/sample/rotation_angle", data=np.linspace(0, 180, 6, endpoint=False))
    project = DataImporter().import_project(path, format="hdf5", storage_mode="ram")
    assert project.vendor_plugin_name == "urct_hdf5"
    assert project.metadata.get("angles").shape == (6,)
    assert project.metadata.get("voxel_size") == 7.5                         # parallel beam: same as the pixel size
    assert project.metadata.get_field("voxel_size").source == "derived"
    project.close()


def test_source_marks_and_extra_metadata_survive(tmp_path):
    path = tmp_path / "x.h5"
    write_minimal(path, geometry_type="parallel", voxel_size=2.0)
    with h5py.File(path, "a") as f:
        f["/entry/geometry/voxel_size"].attrs["source"] = "user"
        f.create_dataset("/entry/extra/beam_current", data=250.0)
    project = DataImporter().import_project(path, format="hdf5", storage_mode="ram")
    assert project.metadata.get_field("voxel_size").source == "user"
    assert project.metadata.get("extra.beam_current") == 250.0
    project.close()


def test_a_newer_major_version_is_refused_and_a_newer_minor_warns(tmp_path):
    path = write_minimal(tmp_path / "n.h5", geometry_type="parallel")
    with h5py.File(path, "a") as f:
        f.attrs["urct_version"] = "1.0"
    assert any("newer" in e for e in fmt.check_file(path).errors)
    with pytest.raises(ValueError, match="standard 1.0"):
        DataImporter().import_project(path, format="hdf5", storage_mode="ram")
    with h5py.File(path, "a") as f:
        f.attrs["urct_version"] = "0.9"
    with pytest.warns(UserWarning, match="newer"):
        DataImporter().import_project(path, format="hdf5", storage_mode="ram").close()


def test_check_file_reports_problems(tmp_path):
    foreign = tmp_path / "f.h5"
    with h5py.File(foreign, "w") as f:
        f.create_dataset("/x", data=1)
    assert "not a URCT HDF5 file" in fmt.check_file(foreign).errors[0]

    cone = write_minimal(tmp_path / "c.h5", geometry_type="cone", angle_span=360.0, voxel_size=1.0)
    report = fmt.check_file(cone)
    assert report.ok and not report.complete
    assert set(report.missing) == {"source_to_detector_distance", "source_to_object_distance"}

    no_data = tmp_path / "nd.h5"
    with h5py.File(no_data, "w") as f:
        fmt.write_header(f, fmt.FILE_VOLUME)
    assert any(fmt.VOLUME in e for e in fmt.check_file(no_data).errors)


def test_a_volume_file_is_not_picked_up_as_projections(tmp_path):
    path = tmp_path / "v.h5"
    with h5py.File(path, "w") as f:
        fmt.write_header(f, fmt.FILE_VOLUME)
        f.create_dataset(fmt.PROJECTIONS, data=np.ones((2, 2, 2)))
    from data_import.vendors.urct_hdf5 import UrctHdf5Plugin

    with h5py.File(path, "r") as f:
        assert not UrctHdf5Plugin().matches(f).matched


def test_interim_layout_files_are_still_read_by_their_own_plugin(tmp_path):
    phantom = generate(shape=(4, 16, 16), n_angles=12, noise_level=0.0, seed=2)
    path = tmp_path / "old.h5"
    write_hdf5(phantom, path)
    project = DataImporter().import_project(path, format="hdf5", storage_mode="ram")
    assert project.vendor_plugin_name == "generic_parallel_synchrotron_hdf5"
    project.close()


def test_completeness_rule():
    assert fmt.is_complete({"geometry_type": "parallel", "angle_span": 180.0, "voxel_size": 1.0})
    assert not fmt.is_complete({"geometry_type": "parallel", "voxel_size": 1.0})
    assert not fmt.is_complete({"geometry_type": "cone", "angles": [0], "voxel_size": 1.0})
    assert fmt.is_complete({"geometry_type": "cone", "angles": [0], "detector_pixel_size": 1.0,
                            "source_to_detector_distance": 3.0, "source_to_object_distance": 2.0})


# -- versioning --------------------------------------------------------------

def test_version_is_one_semver_string_with_a_changelog_entry_and_a_cli_flag(capsys):
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)
    root = Path(__file__).resolve().parents[1]
    assert f"## {__version__}" in (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert 'version = { attr = "core.version.__version__" }' in (root / "pyproject.toml").read_text(encoding="utf-8")
    from orchestration.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--version"])
    assert __version__ in capsys.readouterr().out


def test_software_record_names_the_version_and_given_libraries():
    record = fmt.software_record({"algotom": "1.2"})
    assert record["urct"] == __version__ and record["algotom"] == "1.2" and "python" in record
