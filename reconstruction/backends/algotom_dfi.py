"""
Parallel-beam direct Fourier inversion (DFI) reconstruction backend
wrapping algotom.rec.reconstruction.dfi_reconstruction.

CPU only -- no GPU path in algotom, same as gridrec; actual_device()
is left at ReconstructionBackend's base "always cpu" default.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from reconstruction.backend import Geometry, ReconstructionBackend, register_backend
from reconstruction.backends._algotom_common import algotom_reconstruction


@register_backend
class AlgotomDfiBackend(ReconstructionBackend):
    name = "algotom_dfi"
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

        angles_rad = np.radians(angles)
        n_slices = sinograms.shape[1]
        centers = np.broadcast_to(np.asarray(center, dtype=float), (n_slices,))

        images = [
            rec.dfi_reconstruction(sinograms[:, i, :], float(centers[i]), angles=angles_rad, **options)
            for i in range(n_slices)
        ]
        return np.stack(images)
