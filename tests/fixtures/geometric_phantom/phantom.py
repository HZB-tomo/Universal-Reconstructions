"""
A geometric test phantom: a closed, plastic-filled cylinder in air,
holding a metal pyramid, a cube and a sphere of a less-attenuating
material, and a smaller sphere nested inside the cube.

Distinct from tests/fixtures/realistic_phantom's Shepp-Logan phantom in
how it's built: a few solid geometric primitives rather than a table of
overlapping ellipsoids. Composition is by overwrite (each voxel is
exactly one material), not by summing overlapping densities -- later
shapes win where they overlap earlier ones, which is how inner_sphere
ends up nested inside cube rather than adding to it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Pyramid:
    """Square-based pyramid, axis-aligned, apex toward +z (increasing height index)."""

    material: float
    base_half_width: float
    apex_height: float
    center_xy: tuple[float, float]
    z_base: float


@dataclass
class Cube:
    """Axis-aligned cube."""

    material: float
    half_side: float
    center: tuple[float, float, float]


@dataclass
class Sphere:
    material: float
    radius: float
    center: tuple[float, float, float]


def make_geometric_phantom_volume(
    shape: tuple[int, int, int] = (300, 200, 200),
    cylinder_material: float = 0.3,
    cylinder_margin: float = 2.0,
    pyramid: Optional[Pyramid] = None,
    cube: Optional[Cube] = None,
    sphere: Optional[Sphere] = None,
    inner_sphere: Optional[Sphere] = None,
) -> np.ndarray:
    """
    Render the phantom at `shape` = (height, width, width): a closed
    right cylinder of `cylinder_material`, radius `width / 2 -
    cylinder_margin`, spanning the full height -- the array's own z
    bounds double as the cylinder's closed end caps, nothing extra is
    drawn for them. Air (0.0) outside the cylinder.

    `pyramid`/`cube`/`sphere` default to a metal pyramid and a cube and
    sphere of a less-attenuating material, positioned inside the
    cylinder without overlapping each other or the cylinder wall (with
    a margin) at the default `shape`. `inner_sphere` defaults to a
    lower-density inclusion centered on `cube`, nested inside it.

    Positions/sizes for a non-default `shape` are not rescaled
    automatically; pass explicit Pyramid/Cube/Sphere instances to
    reposition them.

    Returns a float32 density array, shape (height, width, width).
    """
    height, width, width2 = shape
    if width != width2:
        raise ValueError(f"cross-section must be square, got {shape}")

    if pyramid is None:
        pyramid = Pyramid(material=3.0, base_half_width=22.0, apex_height=90.0, center_xy=(0.0, 55.0), z_base=50.0)
    if cube is None:
        cube = Cube(material=1.0, half_side=25.0, center=(-47.6, -27.5, 185.0))
    if sphere is None:
        sphere = Sphere(material=0.8, radius=30.0, center=(47.6, -27.5, 220.0))
    if inner_sphere is None:
        inner_sphere = Sphere(material=0.4, radius=12.0, center=cube.center)

    lin_xy = np.arange(width) - (width - 1) / 2.0
    lin_z = np.arange(height, dtype=np.float64)
    x, y, z = np.meshgrid(lin_xy, lin_xy, lin_z, indexing="ij")  # each (width, width, height)

    density = np.zeros((width, width, height), dtype=np.float32)

    cylinder_radius = width / 2.0 - cylinder_margin
    density[(x**2 + y**2) <= cylinder_radius**2] = cylinder_material

    px0, py0 = pyramid.center_xy
    z_apex = pyramid.z_base + pyramid.apex_height
    in_z = (z >= pyramid.z_base) & (z <= z_apex)
    half_width_at_z = pyramid.base_half_width * np.clip(1.0 - (z - pyramid.z_base) / pyramid.apex_height, 0.0, 1.0)
    pyramid_mask = in_z & (np.abs(x - px0) <= half_width_at_z) & (np.abs(y - py0) <= half_width_at_z)
    density[pyramid_mask] = pyramid.material

    cx0, cy0, cz0 = cube.center
    cube_mask = (
        (np.abs(x - cx0) <= cube.half_side)
        & (np.abs(y - cy0) <= cube.half_side)
        & (np.abs(z - cz0) <= cube.half_side)
    )
    density[cube_mask] = cube.material

    sx0, sy0, sz0 = sphere.center
    sphere_mask = (x - sx0) ** 2 + (y - sy0) ** 2 + (z - sz0) ** 2 <= sphere.radius**2
    density[sphere_mask] = sphere.material

    ix0, iy0, iz0 = inner_sphere.center
    inner_mask = (x - ix0) ** 2 + (y - iy0) ** 2 + (z - iz0) ** 2 <= inner_sphere.radius**2
    density[inner_mask] = inner_sphere.material

    return np.ascontiguousarray(np.moveaxis(density, 2, 0))  # -> (height, width, width)
