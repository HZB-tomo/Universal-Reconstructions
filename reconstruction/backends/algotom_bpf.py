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
    requires = ("algotom",)
    filters = ("ramp", "hann", "hamming", "bartlett", "blackman", "nuttall", "parzen", "triang")
    default_filter = "hann"

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
        if gpu and not rec.cuda.is_available():
            gpu = False
        angles_rad = np.radians(angles)
        n_angles, n_slices, ncol = sinograms.shape
        centers = np.broadcast_to(np.asarray(center, dtype=float), (n_slices,))

        if not gpu:
            # algotom's CPU backprojection has a data race (see _fbp_cpu.py).
            from reconstruction.backends._fbp_cpu import bpf_stack

            return bpf_stack(rec, sinograms, centers, angles_rad, **options)

        from reconstruction.backends._fbp_gpu import bpf_stack_gpu

        return bpf_stack_gpu(rec, sinograms, centers, angles_rad, **options)

    def actual_device(self, device: Optional[str] = None) -> str:
        wants_gpu = True if device is None else (device == "gpu")
        if not wants_gpu:
            return "cpu"
        return "gpu" if algotom_reconstruction(self.name).cuda.is_available() else "cpu"
