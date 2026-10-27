"""
Parallel-beam direct Fourier inversion (DFI) reconstruction backend
wrapping algotom.rec.reconstruction.dfi_reconstruction.

CPU only -- no GPU path in algotom, same as gridrec; actual_device()
is left at ReconstructionBackend's base "always cpu" default.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from reconstruction.backend import Geometry, ReconstructionBackend, register_backend
from reconstruction.backends._algotom_common import algotom_reconstruction
from reconstruction.backends.algotom_fbp import protect_input


@register_backend
class AlgotomDfiBackend(ReconstructionBackend):
    name = "algotom_dfi"
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

        angles_rad = np.radians(angles)
        n_slices = sinograms.shape[1]
        centers = np.broadcast_to(np.asarray(center, dtype=float), (n_slices,))

        sinograms = protect_input(sinograms, options)
        if np.all(centers == centers[0]):
            # One call for the whole stack: algotom builds the polar-to-
            # Cartesian mapping and smoothing window once and runs the
            # slices on parallel threads, instead of redoing the setup and
            # running single-threaded for every slice.
            out = rec.dfi_reconstruction(sinograms, float(centers[0]), angles=angles_rad, **options)
            return np.ascontiguousarray(np.moveaxis(out, 1, 0))

        # Different centers (tilt): the mapping depends on the sinogram
        # shifted to its center, so slices can't share one call -- run
        # them in parallel threads instead of one after another.
        from joblib import Parallel, delayed

        options.setdefault("ncore", 1)  # parallelism is across slices here, not inside a call
        n_jobs = max(1, min(n_slices, (os.cpu_count() or 2) - 1))
        images = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(rec.dfi_reconstruction)(sinograms[:, i, :].copy(), float(centers[i]), angles=angles_rad, **options)
            for i in range(n_slices)
        )
        return np.stack(images)
