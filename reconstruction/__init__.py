"""
Turns normalized sinograms into reconstructed images: one slice,
several, or a whole volume, via a registered backend. Takes plain
arrays, not a ProjectObject.

Public API:
    reconstruct(...)              -> image or image stack, see api.py
    Geometry                      -> parallel/cone geometry description
    ReconstructionBackend         -> the plugin contract
    register_backend, get_backend -> the registry

Importing this package registers the built-in backends (currently
algotom_fbp).
"""
from reconstruction.api import reconstruct
from reconstruction.backend import Geometry, ReconstructionBackend, get_backend, register_backend
from reconstruction import backends  # noqa: F401

__all__ = ["reconstruct", "Geometry", "ReconstructionBackend", "register_backend", "get_backend"]
