"""
reconstruction/ turns normalized sinograms into reconstructed images --
one slice, several, or a whole volume, from whichever backend/algorithm
you choose. Deliberately decoupled from ProjectObject: reconstruct()
takes plain arrays (a sinogram, angles, a center, a geometry), prepared
beforehand by the caller (a pipeline, a notebook, or calibration's own
search loop) -- see reconstruction/api.py's module docstring for why
that separation matters, especially for calibration's use of this
module.

Structure mirrors data_import/{plugin.py, vendors/}: backend.py is the
contract every reconstruction plugin implements plus the registry it
registers with (no signature/auto-detection here, unlike vendor
plugins -- you already know which backend you want, there's nothing in
the data to detect it from); backends/ holds one file per real backend.

Public API:
    reconstruct(...)     -> image or image stack, see api.py
    Geometry              -> parallel/cone geometry description
    ReconstructionBackend  -> the plugin contract
    register_backend, get_backend  -> the registry

Importing this package registers the built-in backends (currently:
algotom_fbp) as a side effect, the same way importing data_import
registers its built-in vendor plugins.
"""
from reconstruction.api import reconstruct
from reconstruction.backend import Geometry, ReconstructionBackend, get_backend, register_backend
from reconstruction import backends  # noqa: F401 -- registers the built-in backends

__all__ = ["reconstruct", "Geometry", "ReconstructionBackend", "register_backend", "get_backend"]
