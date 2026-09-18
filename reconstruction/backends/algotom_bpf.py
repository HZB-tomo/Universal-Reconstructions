"""
Parallel-beam back-projection-filtration (BPF) reconstruction backend
wrapping algotom.rec.reconstruction.bpf_reconstruction -- filters
after backprojecting rather than before, unlike FBP; a different
artifact/accuracy tradeoff for the same underlying geometry.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from reconstruction.backend import Geometry, ReconstructionBackend, register_backend
from reconstruction.backends._algotom_common import algotom_reconstruction


@register_backend
class AlgotomBpfBackend(ReconstructionBackend):
    name = "algotom_bpf"
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
            rec.bpf_reconstruction(sinograms[:, i, :], float(centers[i]), angles=angles_rad, gpu=gpu, **options)
            for i in range(n_slices)
        ]
        return np.stack(images)

    def actual_device(self, device: Optional[str] = None) -> str:
        wants_gpu = True if device is None else (device == "gpu")
        if not wants_gpu:
            return "cpu"
        return "gpu" if algotom_reconstruction(self.name).cuda.is_available() else "cpu"
