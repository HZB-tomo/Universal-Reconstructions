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
from reconstruction.backends._algotom_common import algotom_reconstruction


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
        rec = algotom_reconstruction(self.name)

        gpu = True if device is None else (device == "gpu")

        angles_rad = np.radians(angles)
        n_slices = sinograms.shape[1]
        centers = np.broadcast_to(np.asarray(center, dtype=float), (n_slices,))

        images = [
            rec.fbp_reconstruction(sinograms[:, i, :], float(centers[i]), angles=angles_rad, gpu=gpu, **options)
            for i in range(n_slices)
        ]
        return np.stack(images)

    def actual_device(self, device: Optional[str] = None) -> str:
        wants_gpu = True if device is None else (device == "gpu")
        if not wants_gpu:
            return "cpu"
        # Mirrors fbp_reconstruction()'s own check exactly (it's
        # literally numba.cuda re-exported), so this reports the truth
        # rather than a separate guess that could disagree with it.
        return "gpu" if algotom_reconstruction(self.name).cuda.is_available() else "cpu"
