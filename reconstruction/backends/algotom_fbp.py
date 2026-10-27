"""
Parallel-beam FBP reconstruction backend, built on algotom's ramp filter
and backprojection kernels.

Requires the optional `algotom` package, imported lazily. Raises
ImportError with install instructions if it is not installed.

CPU: algotom's own CPU backprojection has a data race (threads overwrite
each other's sums), so results are wrong by a few percent and differ from
run to run; _fbp_cpu.py replaces it with a race-free kernel.

GPU: one kernel launch per slice (_fbp_gpu.py). algotom's many-slices-
per-launch kernel is slower than that -- it adds into global GPU memory
on every angle -- so it is deliberately not used. The CPU's ramp
filtering of the following slices is overlapped with the GPU's work.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from reconstruction.backend import Geometry, ReconstructionBackend, register_backend
from reconstruction.backends._algotom_common import algotom_reconstruction


def protect_input(sinograms: np.ndarray, options: dict) -> np.ndarray:
    """algotom's reconstructions replace every sinogram value <= 0 with 1
    IN PLACE before taking -log (apply_log=True, the default), silently
    changing the caller's array. Hand it a copy when that would happen."""
    if options.get("apply_log", True) and np.any(sinograms <= 0.0):
        return sinograms.copy()
    return sinograms


@register_backend
class AlgotomFbpBackend(ReconstructionBackend):
    name = "algotom_fbp"
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
            gpu = False  # algotom would warn and fall back anyway; decide once, here

        angles_rad = np.radians(angles)
        n_angles, n_slices, ncol = sinograms.shape
        centers = np.broadcast_to(np.asarray(center, dtype=float), (n_slices,))

        if not gpu:
            # algotom's CPU backprojection has a data race (see _fbp_cpu.py);
            # use the race-free kernel, which handles per-slice centers itself.
            from reconstruction.backends._fbp_cpu import fbp_stack

            return fbp_stack(rec, sinograms, centers, angles_rad, **options)

        from reconstruction.backends._fbp_gpu import fbp_stack_gpu

        return fbp_stack_gpu(rec, sinograms, centers, angles_rad, **options)

    def actual_device(self, device: Optional[str] = None) -> str:
        wants_gpu = True if device is None else (device == "gpu")
        if not wants_gpu:
            return "cpu"
        # Mirrors fbp_reconstruction()'s own check exactly (it's
        # literally numba.cuda re-exported), so this reports the truth
        # rather than a separate guess that could disagree with it.
        return "gpu" if algotom_reconstruction(self.name).cuda.is_available() else "cpu"
