"""
Tests for tests/fixtures/geometric_phantom/. Mirrors
test_realistic_phantom.py's structure: shape/determinism checks, a
round trip through the real generic_parallel_synchrotron plugin (hdf5
and tiff), and containment (nothing drawn outside the cylinder).

Uses a small custom Pyramid/Cube/Sphere layout, not phantom.py's
defaults -- those are sized for the module's own default shape
(300, 200, 200) and are not automatically rescaled for a smaller one
(see make_geometric_phantom_volume()'s docstring).
"""
import numpy as np

from core.data_interface import DataInterface
from data_import.importer import DataImporter
from tests.fixtures.geometric_phantom import Cube, Pyramid, Sphere, generate, make_geometric_phantom_volume, write_hdf5, write_tiff


def _small_shapes():
    pyramid = Pyramid(material=3.0, base_half_width=3.0, apex_height=8.0, center_xy=(0.0, 5.0), z_base=4.0)
    cube = Cube(material=1.0, half_side=3.0, center=(-5.0, -3.0, 14.0))
    sphere = Sphere(material=0.8, radius=3.0, center=(5.0, -3.0, 18.0))
    inner_sphere = Sphere(material=0.4, radius=1.2, center=cube.center)
    return dict(pyramid=pyramid, cube=cube, sphere=sphere, inner_sphere=inner_sphere)


def _small(**overrides):
    params = dict(shape=(24, 32, 32), n_angles=20, n_flats=5, n_darks=5, seed=7, **_small_shapes())
    params.update(overrides)
    return generate(**params)


def test_volume_contains_every_material_and_stays_inside_the_cylinder():
    shape = (24, 32, 32)
    volume = make_geometric_phantom_volume(shape, **_small_shapes())

    assert volume.shape == shape
    values = {round(v, 5) for v in np.unique(volume).tolist()}
    assert values == {0.0, 0.3, 0.4, 0.8, 1.0, 3.0}  # air, plastic, inner_sphere, sphere, cube, pyramid

    height, width, _ = shape
    lin = np.arange(width) - (width - 1) / 2.0
    x, y, z = np.meshgrid(lin, lin, np.arange(height), indexing="ij")
    radius = np.sqrt(x**2 + y**2)
    cylinder_radius = width / 2.0 - 2.0
    outside_but_not_air = (radius > cylinder_radius) & (np.moveaxis(volume, 0, 2) != 0.0)
    assert not outside_but_not_air.any()


def test_generate_shapes_are_consistent():
    phantom = _small()

    assert phantom.volume.shape == (24, 32, 32)
    assert phantom.projections.shape == (20, 24, 32)
    assert phantom.flat_field.shape == (5, 24, 32)
    assert phantom.dark_field.shape == (5, 24, 32)
    assert phantom.angles.shape == (20,)


def test_generate_is_deterministic_with_a_seed():
    a = _small()
    b = _small()

    np.testing.assert_array_equal(a.volume, b.volume)
    np.testing.assert_array_equal(a.projections, b.projections)


def test_cor_deviation_and_tilt_actually_change_the_projections():
    baseline = _small()
    shifted = _small(cor_deviation=1.5)
    tilted = _small(tilt_deg=2.0)

    assert not np.allclose(baseline.projections, shifted.projections)
    assert not np.allclose(baseline.projections, tilted.projections)
    assert shifted.true_center_of_rotation == 1.5
    assert tilted.true_tilt_angle == 2.0


def test_write_hdf5_round_trips_through_the_real_plugin(tmp_path):
    phantom = _small()
    path = tmp_path / "phantom.h5"
    write_hdf5(phantom, path)

    project = DataImporter().import_project(path, format="hdf5", storage_mode="ram")
    di = DataInterface(project)
    np.testing.assert_allclose(di.get_projections(), phantom.projections, rtol=1e-5)
    assert di.get_metadata("geometry_type") == "parallel"


def test_write_tiff_round_trips_through_the_real_plugin(tmp_path):
    phantom = _small()
    out_dir = tmp_path / "phantom_tiff"
    write_tiff(phantom, out_dir)

    project = DataImporter().import_project(out_dir, format="tiff", storage_mode="ram")
    di = DataInterface(project)
    np.testing.assert_allclose(di.get_projections(), phantom.projections, rtol=1e-5)
    assert di.get_metadata("geometry_type") == "parallel"
