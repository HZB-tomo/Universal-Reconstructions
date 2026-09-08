"""
Tests for tests/fixtures/realistic_phantom/ and the
generic_parallel_synchrotron_hdf5 plugin it's paired with. Covers both
consumption paths (to_project_object() in-memory, and write_hdf5() +
DataImporter round-trip), and a quantitative check that the forward
model is actually consistent with preprocessing.normalize() -- not just
that the shapes line up.
"""
import numpy as np

from core.data_interface import DataInterface
from data_import.importer import DataImporter
from preprocessing import build_reference, normalize
from tests.fixtures.realistic_phantom import generate, to_project_object, write_hdf5, write_tiff


def _small(**overrides):
    params = dict(shape=(8, 24, 24), n_angles=30, n_flats=5, n_darks=5, seed=7)
    params.update(overrides)
    return generate(**params)


def test_generate_shapes_are_consistent():
    phantom = _small()

    assert phantom.volume.shape == (8, 24, 24)
    assert phantom.projections.shape == (30, 8, 24)
    assert phantom.flat_field.shape == (5, 8, 24)
    assert phantom.dark_field.shape == (5, 8, 24)
    assert phantom.angles.shape == (30,)


def test_generate_is_deterministic_with_a_seed():
    a = _small()
    b = _small()

    np.testing.assert_array_equal(a.volume, b.volume)
    np.testing.assert_array_equal(a.projections, b.projections)
    np.testing.assert_array_equal(a.flat_field, b.flat_field)


def test_cor_deviation_and_tilt_actually_change_the_projections():
    baseline = _small()
    shifted = _small(cor_deviation=2.5)
    tilted = _small(tilt_deg=3.0)

    assert not np.allclose(baseline.projections, shifted.projections)
    assert not np.allclose(baseline.projections, tilted.projections)
    assert shifted.true_center_of_rotation == 2.5
    assert tilted.true_tilt_angle == 3.0


def test_defective_pixels_and_scratches_show_up_in_the_flat_field():
    clean = _small(n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0)
    flawed = _small(n_scratches=4, n_defective_pixels=6, inhomogeneity_strength=0.0)

    # a clean flat should be much closer to uniform than a flawed one
    assert np.std(flawed.flat_field) > np.std(clean.flat_field)


def test_to_project_object_gives_a_usable_project():
    phantom = _small()
    project = to_project_object(phantom)

    assert project.vendor_plugin_name == "generic_parallel_synchrotron_hdf5"
    assert project.is_ready_for_import()

    di = DataInterface(project)
    np.testing.assert_array_equal(di.get_projections(), phantom.projections)
    np.testing.assert_array_equal(di.get_flats(), phantom.flat_field)
    np.testing.assert_array_equal(di.get_darks(), phantom.dark_field)
    assert di.get_metadata("geometry_type") == "parallel"
    assert di.get_metadata("voxel_size") == phantom.voxel_size

    # computed fields are never blocking, and never faked from ground truth
    assert project.metadata.get_field("center_of_rotation").is_missing
    assert project.metadata.get_field("tilt_angle").is_missing


def test_write_hdf5_round_trips_through_the_real_plugin(tmp_path):
    phantom = _small()
    path = tmp_path / "phantom.h5"
    write_hdf5(phantom, path)

    importer = DataImporter()
    plugin = importer.detect(path, format="hdf5")
    assert plugin.name == "generic_parallel_synchrotron_hdf5"

    project = importer.import_project(path, format="hdf5", storage_mode="ram")
    di = DataInterface(project)
    np.testing.assert_allclose(di.get_projections(), phantom.projections, rtol=1e-5)
    np.testing.assert_allclose(di.get_flats(), phantom.flat_field, rtol=1e-5)
    np.testing.assert_allclose(di.get_darks(), phantom.dark_field, rtol=1e-5)
    np.testing.assert_allclose(di.get_metadata("angles"), phantom.angles, rtol=1e-5)
    assert di.get_metadata("geometry_type") == "parallel"
    # angle_span/detector_pixel_size aren't provided directly by this
    # vendor -- they're derivable (from angles, and from voxel_size,
    # respectively), but raw import only reconciles, it doesn't run
    # completion -- so they're correctly still missing at this point.
    assert set(project.missing_soft_fields()) == {"angle_span", "detector_pixel_size"}


def test_write_tiff_round_trips_through_the_real_plugin(tmp_path):
    phantom = _small()
    out_dir = tmp_path / "phantom_tiff"
    write_tiff(phantom, out_dir)

    importer = DataImporter()
    plugin = importer.detect(out_dir, format="tiff")
    assert plugin.name == "generic_parallel_synchrotron_tiff"

    project = importer.import_project(out_dir, format="tiff", storage_mode="ram")
    di = DataInterface(project)
    np.testing.assert_allclose(di.get_projections(), phantom.projections, rtol=1e-5)
    np.testing.assert_allclose(di.get_flats(), phantom.flat_field, rtol=1e-5)
    np.testing.assert_allclose(di.get_darks(), phantom.dark_field, rtol=1e-5)
    np.testing.assert_allclose(di.get_metadata("angles"), phantom.angles, rtol=1e-5)
    assert di.get_metadata("geometry_type") == "parallel"
    assert di.get_metadata("voxel_size") == phantom.voxel_size


def test_write_tiff_disk_mode_never_loads_the_full_stack_eagerly(tmp_path):
    from core.handles import TiffDiskHandle

    phantom = _small()
    out_dir = tmp_path / "phantom_tiff"
    write_tiff(phantom, out_dir)

    project = DataImporter().import_project(out_dir, format="tiff", storage_mode="disk")

    assert isinstance(project.data_handles["projections"], TiffDiskHandle)
    assert project.data_handles["projections"].shape == phantom.projections.shape


def test_forward_model_is_consistent_with_normalize(tmp_path):
    """
    The real point of this generator: normalize() on noiseless output
    should recover the ideal transmission almost exactly, since the
    forward model is built as normalize()'s exact inverse.
    """
    phantom = _small(noise_level=0.0, n_scratches=0, n_defective_pixels=0)
    path = tmp_path / "phantom.h5"
    write_hdf5(phantom, path)

    project = DataImporter().import_project(path, format="hdf5", storage_mode="ram")
    reference = build_reference(project)
    recovered = normalize(phantom.projections, reference)

    expected_transmission = np.exp(-2.0 * project_line_integrals(phantom))
    np.testing.assert_allclose(recovered, expected_transmission, rtol=1e-3, atol=1e-3)


def project_line_integrals(phantom):
    """Re-derive the ideal (pre-detector) line integrals from the phantom's own volume/angles/geometry, for the consistency check above."""
    from tests.fixtures.realistic_phantom.simulate import apply_cor_and_tilt, project_parallel

    raw = project_parallel(phantom.volume, phantom.angles)
    return apply_cor_and_tilt(raw, phantom.true_center_of_rotation, phantom.true_tilt_angle)


def test_does_not_get_confused_with_hypothetical_synchrotron(tmp_path):
    """Two similarly-shaped synchrotron-style plugins, distinguished only
    by their root attribute -- confirm detection actually tells them apart."""
    phantom = _small()
    path = tmp_path / "phantom.h5"
    write_hdf5(phantom, path)

    plugin = DataImporter().detect(path, format="hdf5")
    assert plugin.name == "generic_parallel_synchrotron_hdf5"
