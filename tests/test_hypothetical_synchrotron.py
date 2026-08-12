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
    # input for it -- only geometry_type is still missing.
    assert project.needs_user_input()
    assert project.missing_soft_fields() == ["geometry_type"]

    di = DataInterface(project)

    proj = di.get_projections()
    np.testing.assert_allclose(proj, projections, rtol=1e-4, atol=1e-3)

    loaded_flats = di.get_flats()
    np.testing.assert_allclose(loaded_flats, flats, rtol=1e-4, atol=1e-3)

    np.testing.assert_allclose(di.get_metadata("angles"), angles)
    assert di.get_metadata("voxel_size") == 1.1
    assert di.get_metadata("energy") == 25000.0
