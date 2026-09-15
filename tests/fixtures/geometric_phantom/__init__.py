"""
A geometric test phantom, distinct from tests/fixtures/realistic_phantom's
Shepp-Logan phantom: a closed, plastic-filled cylinder in air, holding a
metal pyramid, a cube and a sphere of a less-attenuating material, and a
smaller sphere nested inside the cube.

Shares its detector forward model (flat/dark simulation, COR/tilt
injection, noise) with realistic_phantom -- only the object's geometry
(phantom.py) differs, so write_hdf5()/write_tiff() are realistic_phantom's
own, reused here rather than duplicated.

Public API:
    generate(...)              -> GeometricPhantomProject
    write_hdf5(phantom, path)  -> None
    write_tiff(phantom, dir)   -> None
    Pyramid, Cube, Sphere      -> shape parameter dataclasses, see phantom.py
    make_geometric_phantom_volume(...) -> the object alone, no detector simulation
"""
from tests.fixtures.geometric_phantom.generate import GeometricPhantomProject, generate
from tests.fixtures.geometric_phantom.phantom import Cube, Pyramid, Sphere, make_geometric_phantom_volume
from tests.fixtures.realistic_phantom.generate import write_hdf5, write_tiff

__all__ = [
    "GeometricPhantomProject", "generate", "write_hdf5", "write_tiff",
    "Pyramid", "Cube", "Sphere", "make_geometric_phantom_volume",
]
