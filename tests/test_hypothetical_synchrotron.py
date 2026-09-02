"""
Tests for HypotheticalSynchrotronHdf5Plugin: confirms it's correctly
detected (and doesn't get confused with synthetic_hdf5 or generic_hdf5),
and that import produces correct data/metadata, including the
vendor-specific metadata tiering (voxel_size arrives from the file here,
unlike the synthetic/generic plugins where it's always missing).
"""
import numpy as np

from data_import.importer import DataImporter
from core.data_interface import DataInterface
from tests.fixtures.generate_hypothetical_synchrotron import generate


def test_detects_hypothetical_synchrotron_plugin(tmp_path):
    path = tmp_path / "beamline_scan.h5"
    generate(path, shape=(24, 24, 32), n_angles=10, n_flats=4, seed=7)

    importer = DataImporter()
    plugin = importer.detect(path, format="hdf5")
    assert plugin.name == "hypothetical_synchrotron_hdf5"


def test_import_hypothetical_synchrotron(tmp_path):
    path = tmp_path / "beamline_scan.h5"
    volume, projections, flats, angles = generate(
        path, shape=(24, 24, 32), n_angles=10, n_flats=4, seed=7
    )

    importer = DataImporter()
    project = importer.import_project(path, format="hdf5", storage_mode="ram")

    assert project.vendor_plugin_name == "hypothetical_synchrotron_hdf5"
    assert project.is_ready_for_import()

    # voxel_size came straight from the file for this vendor, so unlike
    # the synthetic/generic plugins, the project should NOT need user
    # input for it. geometry_type is missing because the plugin doesn't
    # know it; angle_span and detector_pixel_size are missing because
    # core/known_fields.reconcile() adds them for every project this
    # plugin never mentions them for (see importer.py's import_project()).
    # source_to_detector_distance/source_to_object_distance are cone-only
    # and correctly absent here since geometry_type isn't resolved yet.
    assert project.needs_user_input()
    assert set(project.missing_soft_fields()) == {"geometry_type", "angle_span", "detector_pixel_size"}

    di = DataInterface(project)

    proj = di.get_projections()
    np.testing.assert_allclose(proj, projections, rtol=1e-4, atol=1e-3)

    loaded_flats = di.get_flats()
    np.testing.assert_allclose(loaded_flats, flats, rtol=1e-4, atol=1e-3)

    np.testing.assert_allclose(di.get_metadata("angles"), angles)
    assert di.get_metadata("voxel_size") == 1.1
    assert di.get_metadata("energy") == 25000.0
