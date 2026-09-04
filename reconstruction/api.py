"""
reconstruct(): the one public entry point for turning sinogram(s) into
image(s). Deliberately takes plain arrays, not a ProjectObject -- the
caller (a pipeline, a notebook, or calibration's own search loop)
prepares whatever data this needs beforehand, the same way it already
calls preprocessing.normalize_rows() itself rather than reconstruct()
reaching into a project on its own.

Why this matters specifically for calibration: a center-of-rotation
search doesn't reconstruct "row R of some project" over and over --
it reconstructs many CANDIDATE variants of the same row (each one
pre-shifted by a different trial offset, exactly like
oldscripts/BAMline_recon_v0p2.py's find_CoR() already does), stacks
them, and reconstructs the whole stack in one call. That data has no
natural "give me project row R" expression -- it's a purpose-built
array the search loop constructs for its own purposes. A ProjectObject-
aware reconstruct() couldn't serve that; a plain array-in-array-out one
does, for free, with no special "candidate" concept anywhere in this
module -- reconstruct() never knows or cares whether the sinograms it's
handed are different physical rows or shifted copies of the same one.

Units and dtype are also handled here, centrally, once, rather than by
each backend -- see the "Units" note below and doc/RECONSTRUCTION.md.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from reconstruction.backend import Geometry, ReconstructionBackend, get_backend


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

        reconstruct(one_sinogram, angles, center, geometry, "algotom_fbp")            -> one 2D image
        reconstruct(stack_of_sinograms, angles, center, geometry, "algotom_fbp")      -> a matching 3D stack

    Which shape you get back matches which shape you passed in --
    numpy-indexing style, not two differently-named functions: pass a
    single (n_angles, width) sinogram for one image back, or a
    (n_angles, n_slices, width) stack -- angle-first, exactly what
    preprocessing.normalize_rows() already returns, e.g.
    `normalize_rows(project, some_rows, reference)` can be handed
    straight to this function with no reshaping -- for a matching
    (n_slices, height, width) image stack back.

    `backend` selects which plugin does the work -- a name to look up
    via get_backend(), or a ReconstructionBackend instance directly.
    Always explicit; nothing here auto-detects a backend the way
    vendor import plugins auto-detect a file's vendor.

    `center` is a scalar (shared by every sinogram in the stack) or an
    array-like with one value per sinogram -- see backend.py's
    ReconstructionBackend.reconstruct() docstring.

    `device`/`**options` are passed straight through to the backend --
    see that backend's own documentation for what it accepts.

    Units: every backend returns raw attenuation coefficients in "per
    pixel" units (the natural output of any FBP/iterative algorithm,
    which has no idea what a pixel physically measures). This function
    converts that to a real physical unit itself, the only way a caller
    could actually want it -- there's no meaningful "which unit would
    you like" choice to offer: if `geometry.detector_pixel_size` is
    given (this project always stores it in micrometers -- see
    `Geometry`), the result is divided by it and comes back as
    attenuation in inverse micrometers (1/um). If `detector_pixel_size`
    is None, there's nothing to convert by, so the result stays in
    "per pixel" units (1/px) -- not physically comparable across
    datasets with different pixel sizes. See doc/RECONSTRUCTION.md's
    "Units" section for the full explanation.

    The returned array is always float32, regardless of what the
    backend itself returns internally -- verified against algotom's own
    fbp_reconstruction, which already always returns float32 regardless
    of input precision. float32 is standard practice for tomography
    specifically because most GPUs (consumer and datacenter alike) have
    much lower float64 throughput than float32, and float32 already has
    far more precision than detector data or attenuation coefficients
    need -- so this is enforced once, here, rather than left to each
    backend to get right.
    """
    backend_instance = get_backend(backend) if isinstance(backend, str) else backend

    if geometry.geometry_type not in backend_instance.supported_geometries:
        raise ValueError(
            f"Backend '{backend_instance.name}' doesn't support geometry_type={geometry.geometry_type!r} "
            f"(supports: {sorted(backend_instance.supported_geometries)})"
        )

    sinogram = np.asarray(sinogram)
    was_single = sinogram.ndim == 2
    # A bare (n_angles, width) sinogram is wrapped to (n_angles, 1, width) --
    # inserting the new axis at position 1, not 0, to match the
    # angle-first stack convention (see this function's docstring).
    stack = sinogram[:, np.newaxis, :] if was_single else sinogram

    images = backend_instance.reconstruct(stack, np.asarray(angles), center, geometry, device, **options)

    # "per pixel" -> real attenuation coefficient (1/um) -- see this
    # function's "Units" docstring section above.
    if geometry.detector_pixel_size is not None:
        images = images / np.float32(geometry.detector_pixel_size)
    images = np.asarray(images, dtype=np.float32)

    return images[0] if was_single else images
