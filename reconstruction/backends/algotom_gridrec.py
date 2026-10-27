"""
Parallel-beam gridrec (Fourier-gridding) reconstruction backend
wrapping algotom.rec.reconstruction.gridrec_reconstruction.

CPU only -- algotom's gridrec has no GPU path at all (unlike
fbp_reconstruction/bpf_reconstruction); actual_device() is left at
ReconstructionBackend's base "always cpu" default rather than
overridden.

Typically much faster than FBP for large angle counts (synchrotron-
scale scans), via a Fourier-domain interpolation instead of a direct
filtered backprojection -- a different accuracy/speed tradeoff, not a
strictly better one.

algotom doesn't actually implement gridrec itself: gridrec_reconstruction()
is a thin wrapper that defers to TomoPy's own implementation, and
raises a plain ValueError (not ImportError) if TomoPy isn't installed
-- caught and re-raised here as ImportError for the same clear,
consistent "not installed" signal every other backend gives. TomoPy is
conda-only, unlike every other dependency in this project (confirmed
separately: no pip wheel exists at all), so this backend needs conda
specifically, not `pip install`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from reconstruction.backend import Geometry, ReconstructionBackend, register_backend
from reconstruction.backends._algotom_common import algotom_reconstruction


@register_backend
class AlgotomGridrecBackend(ReconstructionBackend):
    name = "algotom_gridrec"
    supported_geometries = {"parallel"}
    install_hint = "Install with: conda install -c conda-forge tomopy (no pip wheel). The 'gridrec' entry needs no conda."
    requires = ("algotom", "tomopy")   # tomopy has no pip wheel (conda only); see the module docstring

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

        try:
            images = [
                rec.gridrec_reconstruction(sinograms[:, i, :], float(centers[i]), angles=angles_rad, **options)
                for i in range(n_slices)
            ]
        except ValueError as e:
            if "Tomopy" not in str(e):
                raise
            raise ImportError(
                "The 'algotom_gridrec' reconstruction backend requires the 'tomopy' package, which isn't installed. "
                "TomoPy has no pip wheel -- install it with conda: conda install -c conda-forge tomopy"
            ) from e
        return np.stack(images)
