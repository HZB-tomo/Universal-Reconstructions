"""
End-to-end tests: actually import the synthetic hdf5 and tiff fixtures,
and confirm the resulting ProjectObject + DataInterface behave exactly
as expected -- correct plugin selected, correct data loaded back, RAM
vs disk modes both work, and materializing disk -> RAM works.
"""
import numpy as np

from data_import.importer import DataImporter
from core.data_interface import DataInterface
from core.handles import RamHandle, Hdf5DiskHandle


def test_import_hdf5_ram(synthetic_data):
    importer = DataImporter()
    project = importer.import_project(synthetic_data["hdf5_path"], format="hdf5", storage_mode="ram")

    assert project.vendor_plugin_name == "synthetic_hdf5"
    assert project.is_ready_for_import()
    assert project.needs_user_input()  # voxel_size/geometry_type are still soft-missing
    assert isinstance(project.data_handles["projections"], RamHandle)

    di = DataInterface(project)
    proj = di.get_projections()
    assert proj.shape == synthetic_data["projections"].shape
    np.testing.assert_allclose(proj, synthetic_data["projections"], rtol=1e-4, atol=1e-3)

    angles = di.get_metadata("angles")
    np.testing.assert_allclose(angles, synthetic_data["angles"])


def test_import_hdf5_disk(synthetic_data):
    importer = DataImporter()
    project = importer.import_project(synthetic_data["hdf5_path"], format="hdf5", storage_mode="disk")

    assert isinstance(project.data_handles["projections"], Hdf5DiskHandle)
    di = DataInterface(project)
    # slicing should work without loading the whole array into memory
    first_angle = di.get_projections(index=0)
    assert first_angle.shape == synthetic_data["projections"].shape[1:]

    project.close()


def test_import_hdf5_disk_materialize_to_ram(synthetic_data):
    importer = DataImporter()
    project = importer.import_project(synthetic_data["hdf5_path"], format="hdf5", storage_mode="disk")
    di = DataInterface(project)

    di.materialize_to_ram()

    assert project.storage_mode == "ram"
    assert isinstance(project.data_handles["projections"], RamHandle)


def test_import_tiff_ram(synthetic_data):
    importer = DataImporter()
    project = importer.import_project(synthetic_data["tiff_path"], format="tiff", storage_mode="ram")

    assert project.vendor_plugin_name == "synthetic_tiff"
    assert isinstance(project.data_handles["projections"], RamHandle)

    di = DataInterface(project)
    proj = di.get_projections()
    assert proj.shape == synthetic_data["projections"].shape
    np.testing.assert_allclose(proj, synthetic_data["projections"], rtol=1e-4, atol=1e-3)

    assert di.get_metadata("operator") == "synthetic data"


def test_import_tiff_disk_mode_rejected(synthetic_data):
    import pytest

    importer = DataImporter()
    with pytest.raises(ValueError):
        importer.import_project(synthetic_data["tiff_path"], format="tiff", storage_mode="disk")
