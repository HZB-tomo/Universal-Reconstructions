"""
Ties phantom.py (this phantom's geometry) and
tests/fixtures/realistic_phantom/simulate.py (the detector forward
model, shared with the Shepp-Logan phantom -- flat/dark simulation,
COR/tilt injection, noise) into one entry point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tests.fixtures.geometric_phantom.phantom import Cube, Pyramid, Sphere, make_geometric_phantom_volume
from tests.fixtures.realistic_phantom.simulate import (
    apply_cor_and_tilt,
    attenuation_scale_for_min_transmission,
    make_detector_patterns,
    project_parallel,
    synthesize_raw,
)


@dataclass
class GeometricPhantomProject:
    """
    Same shape as tests.fixtures.realistic_phantom.PhantomProject
    (projections/flat_field/dark_field/angles/voxel_size), kept as a
    separate class since the two phantoms' geometry parameters differ
    and shouldn't be confused. write_hdf5()/write_tiff() only touch
    those shared fields, so realistic_phantom's own are reused as-is,
    not duplicated here.
    """

    volume: np.ndarray  # ground-truth density, shape (height, width, width)
    projections: np.ndarray  # simulated raw detector readings, (n_angles, height, width)
    flat_field: np.ndarray
    dark_field: np.ndarray
    angles: np.ndarray  # degrees
    true_center_of_rotation: float  # pixels, deviation from image center at the middle row
    true_tilt_angle: float  # degrees
    voxel_size: float


def generate(
    shape: tuple[int, int, int] = (300, 200, 200),
    n_angles: int = 300,
    n_flats: int = 10,
    n_darks: int = 10,
    cor_deviation: float = 0.0,
    tilt_deg: float = 0.0,
    attenuation_scale: Optional[float] = None,
    target_min_transmission: float = 0.14,
    cylinder_material: float = 0.3,
    pyramid: Optional[Pyramid] = None,
    cube: Optional[Cube] = None,
    sphere: Optional[Sphere] = None,
    inner_sphere: Optional[Sphere] = None,
    flat_intensity: float = 1.0,
    dark_level: float = 0.01,
    inhomogeneity_strength: float = 0.05,
    n_scratches: int = 3,
    scratch_strength: float = 0.15,
    n_defective_pixels: int = 5,
    noise_level: float = 0.0,
    voxel_size: float = 1.0,
    seed: Optional[int] = None,
) -> GeometricPhantomProject:
    """
    Simulate one CT acquisition of the geometric phantom (cylinder,
    pyramid, cube, sphere, inner_sphere -- see phantom.py's
    make_geometric_phantom_volume()). Every other parameter forwards to
    the same forward model realistic_phantom.generate() uses; see that
    module's docstring, simulate.py, and doc/REALISTIC_PHANTOM.md.

    `attenuation_scale`: None (default) solves for it via
    attenuation_scale_for_min_transmission(), so the single
    most-attenuating ray ends up at `target_min_transmission` of the
    free beam, matching how a real acquisition's exposure/filtration is
    actually set up. See realistic_phantom.generate()'s docstring and
    simulate.py's attenuation_scale_for_min_transmission().
    """
    rng = np.random.default_rng(seed)
    height, width = shape[0], shape[1]

    volume = make_geometric_phantom_volume(
        shape, cylinder_material=cylinder_material,
        pyramid=pyramid, cube=cube, sphere=sphere, inner_sphere=inner_sphere,
    )
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    line_integrals = project_parallel(volume, angles)
    line_integrals = apply_cor_and_tilt(line_integrals, cor_deviation, tilt_deg)

    if attenuation_scale is None:
        attenuation_scale = attenuation_scale_for_min_transmission(line_integrals, target_min_transmission)

    gain, dark_offset = make_detector_patterns(
        (height, width), inhomogeneity_strength, n_scratches, scratch_strength, n_defective_pixels, dark_level, rng
    )
    projections, flat_field, dark_field = synthesize_raw(
        line_integrals, gain, dark_offset, flat_intensity, attenuation_scale, n_flats, n_darks, noise_level, rng
    )

    return GeometricPhantomProject(
        volume=volume, projections=projections, flat_field=flat_field, dark_field=dark_field,
        angles=angles, true_center_of_rotation=cor_deviation, true_tilt_angle=tilt_deg, voxel_size=voxel_size,
    )
