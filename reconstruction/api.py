"""
reconstruct(): the public entry point for turning sinogram(s) into
image(s). Takes plain arrays, not a ProjectObject. The caller prepares
whatever data this needs beforehand, the same way it calls
preprocessing.normalize_rows() itself.

Unit conversion and dtype normalization also happen here; see the
"Units" section of reconstruct()'s docstring and doc/RECONSTRUCTION.md.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from reconstruction.backend import Geometry, ReconstructionBackend, get_backend

_UM_PER_CM = 1.0e4


def reconstruct(
    sinogram: np.ndarray,
    angles: np.ndarray,
    center,
    geometry: Geometry,
    backend: Union[str, ReconstructionBackend],
    device: Optional[str] = None,
    **options,
) -> np.ndarray:
    """
    Reconstruct one sinogram or a stack of them.

        reconstruct(one_sinogram, angles, center, geometry, "algotom_fbp")       -> one 2D image
        reconstruct(stack_of_sinograms, angles, center, geometry, "algotom_fbp") -> a matching 3D stack

    A single (n_angles, width) sinogram returns a single (height, width)
    image. A (n_angles, n_slices, width) stack, angle-first, returns a
    matching (n_slices, height, width) image stack. This is the same
    shape preprocessing.normalize_rows() returns, so its result can be
    passed directly.

    `backend` is a registered backend name or a ReconstructionBackend
    instance.

    `center` is a scalar, shared by every sinogram in the stack, or an
    array-like with one value per sinogram. See
    ReconstructionBackend.reconstruct() in backend.py.

    `device`/`**options` are passed through to the backend.

    Units: a backend returns raw attenuation coefficients in "per
    pixel" units. If `geometry.detector_pixel_size` is set (stored in
    micrometers, see Geometry), the result is divided by it and scaled
    to inverse centimeters (1/cm) -- the conventional unit for a linear
    attenuation coefficient, not inverse micrometers, which is what
    dividing by a micrometer-scale pixel size gives directly (see
    _UM_PER_CM). If `detector_pixel_size` is None, the result stays in
    "per pixel" units (1/px), not physically comparable across datasets
    with different pixel sizes. See doc/RECONSTRUCTION.md's "Units" section.

    The returned array is always float32, regardless of what the
    backend returns internally.
    """
    backend_instance = get_backend(backend) if isinstance(backend, str) else backend

    if geometry.geometry_type not in backend_instance.supported_geometries:
        raise ValueError(
            f"Backend '{backend_instance.name}' doesn't support geometry_type={geometry.geometry_type!r} "
            f"(supports: {sorted(backend_instance.supported_geometries)})"
        )

    name = options.get("filter_name")
    if name is not None and backend_instance.filters and name not in backend_instance.filters:
        raise ValueError(
            f"Backend '{backend_instance.name}' has no filter {name!r} (choose from {list(backend_instance.filters)})"
        )

    sinogram = np.asarray(sinogram)
    was_single = sinogram.ndim == 2
    stack = sinogram[:, np.newaxis, :] if was_single else sinogram

    images = backend_instance.reconstruct(stack, np.asarray(angles), center, geometry, device, **options)

    if geometry.detector_pixel_size is not None:
        images = images / np.float32(geometry.detector_pixel_size) * np.float32(_UM_PER_CM)
    images = np.asarray(images, dtype=np.float32)

    return images[0] if was_single else images


def resolve_device(backend: Union[str, ReconstructionBackend], device: Optional[str] = None) -> str:
    """
    "gpu" or "cpu": the device a reconstruct() call with this same
    `backend`/`device` will actually use, accounting for real hardware
    availability -- not just the `device` hint, which for most backends
    defaults to "try GPU, fall back to CPU" and so doesn't say what
    actually happened on its own. See ReconstructionBackend.actual_device().
    """
    backend_instance = get_backend(backend) if isinstance(backend, str) else backend
    return backend_instance.actual_device(device)
