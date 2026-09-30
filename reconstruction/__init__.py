"""
Turns normalized sinograms into reconstructed images: one slice,
several, or a whole volume, via a registered backend. Takes plain
arrays, not a ProjectObject.

Public API:
    reconstruct(...)              -> image or image stack, see api.py
    resolve_device(...)           -> "gpu"/"cpu" actually used, see api.py
    reconstruct_rotated(...)      -> reconstruct(), rotated about the scan axis, see rotation.py
    supports_exact_rotation(...)  -> whether that rotation can be exact for a given scan, see rotation.py
    to_uint16(...)                 -> dtype/clip conversion, see postprocess.py
    save_volume(...)              -> write a reconstructed volume to disk, see export.py
    StreamingVolumeWriter         -> incremental, chunk-at-a-time counterpart to save_volume(), see export.py
    Geometry                      -> parallel/cone geometry description
    ReconstructionBackend         -> the plugin contract
    register_backend, get_backend -> the registry

Importing this package registers the built-in backends (currently
algotom_fbp).
"""
from reconstruction.api import reconstruct, resolve_device
from reconstruction.backend import Geometry, ReconstructionBackend, get_backend, register_backend
from reconstruction.export import StreamingVolumeWriter, save_volume
from reconstruction.postprocess import to_uint16
from reconstruction.rotation import reconstruct_rotated, supports_exact_rotation
from reconstruction import backends  # noqa: F401

__all__ = [
    "reconstruct", "resolve_device", "reconstruct_rotated", "supports_exact_rotation",
    "to_uint16", "save_volume", "StreamingVolumeWriter",
    "Geometry", "ReconstructionBackend", "register_backend", "get_backend",
]
