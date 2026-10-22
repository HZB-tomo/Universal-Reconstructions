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
from reconstruction.backends.algotom_fbp import gpu_batch_size, gpu_free_bytes, plan_batches, protect_input


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
        if gpu and not rec.cuda.is_available():
            gpu = False
        angles_rad = np.radians(angles)
        n_angles, n_slices, ncol = sinograms.shape
        centers = np.broadcast_to(np.asarray(center, dtype=float), (n_slices,))

        if not gpu:
            # algotom's CPU backprojection has a data race (see _fbp_cpu.py).
            from reconstruction.backends._fbp_cpu import bpf_stack

            return bpf_stack(rec, sinograms, centers, angles_rad, **options)

        # GPU: one batched call for slices sharing a center (one kernel launch
        # for all of them), otherwise per slice with the ramp window built once.
        sinograms = protect_input(sinograms, options)
        if np.all(centers == centers[0]):
            batch = gpu_batch_size(n_angles, ncol, gpu_free_bytes(rec))
            parts = []
            for chunk in plan_batches(n_slices, batch):
                out = rec.bpf_reconstruction(
                    sinograms[:, chunk, :], float(centers[0]), angles=angles_rad, gpu=True, **options,
                )
                parts.append(np.ascontiguousarray(np.moveaxis(out, 1, 0)))
            return parts[0] if len(parts) == 1 else np.concatenate(parts)

        if "ramp_win" not in options:
            pad = options.get("pad")
            pad = min(int(0.15 * ncol), 150) if pad is None else pad
            options["ramp_win"] = rec.make_circular_ramp_window(ncol + 2 * pad, options.get("filter_name", "hann"))
        return np.stack([
            rec.bpf_reconstruction(sinograms[:, i, :], float(centers[i]), angles=angles_rad, gpu=True, **options)
            for i in range(n_slices)
        ])

    def actual_device(self, device: Optional[str] = None) -> str:
        wants_gpu = True if device is None else (device == "gpu")
        if not wants_gpu:
            return "cpu"
        return "gpu" if algotom_reconstruction(self.name).cuda.is_available() else "cpu"
