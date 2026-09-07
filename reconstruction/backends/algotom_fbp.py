"""
Parallel-beam FBP reconstruction backend wrapping
algotom.rec.reconstruction.fbp_reconstruction.

Requires the optional `algotom` package, imported lazily. Raises
ImportError with install instructions if it is not installed.

algotom's fbp_reconstruction() reconstructs one sinogram at a time;
this backend loops over the requested stack.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from reconstruction.backend import Geometry, ReconstructionBackend, register_backend


def _algotom_reconstruction():
    try:
        import algotom.rec.reconstruction as rec
    except ImportError as e:
        raise ImportError(
            "The 'algotom_fbp' reconstruction backend requires the 'algotom' package, which isn't installed. "
            "Install it with: pip install algotom"
        ) from e
    return rec


@register_backend
class AlgotomFbpBackend(ReconstructionBackend):
    name = "algotom_fbp"
    supported_geometries = {"parallel"}

    def reconstruct(
        self,
        sinograms: np.ndarray,
        angles: np.ndarray,
        center,
        geometry: Geometry,
        device: Optional[str] = None,
        **options,
    ) -> np.ndarray:
        rec = _algotom_reconstruction()

        gpu = True if device is None else (device == "gpu")

        angles_rad = np.radians(angles)
        n_slices = sinograms.shape[1]
        centers = np.broadcast_to(np.asarray(center, dtype=float), (n_slices,))

        images = [
            rec.fbp_reconstruction(sinograms[:, i, :], float(centers[i]), angles=angles_rad, gpu=gpu, **options)
            for i in range(n_slices)
        ]
        return np.stack(images)
