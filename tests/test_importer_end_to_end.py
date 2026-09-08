"""
End-to-end tests: actually import the synthetic hdf5 and tiff fixtures,
and confirm the resulting ProjectObject + DataInterface behave exactly
as expected -- correct plugin selected, correct data loaded back, RAM
vs disk modes both work, and materializing disk -> RAM works.
"""
import numpy as np

from data_import.importer import DataImporter
from core.data_interface import DataInterface
from core.handles import RamHandle, Hdf5DiskHandle, TiffDiskHandle


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


def test_import_tiff_disk(synthetic_data):
    importer = DataImporter()
    project = importer.import_project(synthetic_data["tiff_path"], format="tiff", storage_mode="disk")

    assert isinstance(project.data_handles["projections"], TiffDiskHandle)
    di = DataInterface(project)
    # slicing should work without loading the whole stack into memory
    first_angle = di.get_projections(index=0)
    assert first_angle.shape == synthetic_data["projections"].shape[1:]

    proj = di.get_projections()
    assert proj.shape == synthetic_data["projections"].shape
    np.testing.assert_allclose(proj, synthetic_data["projections"], rtol=1e-4, atol=1e-3)


def test_import_tiff_disk_supports_multi_axis_tuple_indexing(synthetic_data):
    """
    preprocessing.normalize_rows() indexes get_projections() with a
    3-tuple (slice(None), rows, slice(None)): every angle, a row
    subset, every column. h5py.Dataset.__getitem__ handles this
    natively for Hdf5DiskHandle; TiffDiskHandle has to do it itself
    since it only has a plain list of file paths, not an array.
    """
    importer = DataImporter()
    project = importer.import_project(synthetic_data["tiff_path"], format="tiff", storage_mode="disk")
    di = DataInterface(project)

    rows = slice(1, 3)
    subset = di.get_projections(index=(slice(None), rows, slice(None)))

    expected = synthetic_data["projections"][:, rows, :]
    assert subset.shape == expected.shape
    np.testing.assert_allclose(subset, expected, rtol=1e-4, atol=1e-3)


def test_import_tiff_disk_shape_matches_the_full_stack(synthetic_data):
    importer = DataImporter()
    project = importer.import_project(synthetic_data["tiff_path"], format="tiff", storage_mode="disk")

    assert project.data_handles["projections"].shape == synthetic_data["projections"].shape


def test_import_tiff_disk_materialize_to_ram(synthetic_data):
    importer = DataImporter()
    project = importer.import_project(synthetic_data["tiff_path"], format="tiff", storage_mode="disk")
    di = DataInterface(project)

    di.materialize_to_ram()

    assert project.storage_mode == "ram"
    assert isinstance(project.data_handles["projections"], RamHandle)
    np.testing.assert_allclose(
        project.data_handles["projections"].get(), synthetic_data["projections"], rtol=1e-4, atol=1e-3
    )


def test_import_reconciles_fields_no_plugin_mentions(synthetic_data):
    """
    angle_span is a core/known_fields.py entry the synthetic plugin
    never mentions at all -- it should still show up as soft-missing
    after import, proving import_project() actually calls
    known_fields.reconcile() rather than relying on the plugin to
    self-report every field the software depends on.
    """
    importer = DataImporter()
    project = importer.import_project(synthetic_data["hdf5_path"], format="hdf5", storage_mode="ram")

    assert "angle_span" in project.missing_soft_fields()
