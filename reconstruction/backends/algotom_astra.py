"""
Reconstruction backends wrapping
algotom.rec.reconstruction.astra_reconstruction -- algotom's own
wrapper around the ASTRA Toolbox's 2D algorithms. Requires BOTH the
optional 'algotom' package AND 'astra-toolbox', a further, separate
optional dependency; astra_reconstruction() itself lazily imports
astra and raises a clear ValueError if it's missing, so that isn't
duplicated here.

Five algorithms, each its own backend so it shows up as its own choice
in the GUI's reconstruction-algorithm list rather than a hidden option
string:
    algotom_bp    -- plain backprojection, no filtering (mostly a
                      diagnostic/comparison baseline, not for real use)
    algotom_sirt  -- Simultaneous Iterative Reconstruction Technique
    algotom_sart  -- Simultaneous Algebraic Reconstruction Technique
    algotom_art   -- Algebraic Reconstruction Technique (Kaczmarz)
    algotom_cgls  -- Conjugate Gradient Least Squares

All but BP are iterative: slower than FBP/gridrec/BPF/DFI, but handle
noisy, sparse-view, or limited-angle data better. Iteration count is
`num_iter` in **options -- algotom's own default is 1, almost always
too few for a real reconstruction; pass a real value.

`num_iter` means something very different for ART than for SIRT/SART/
CGLS, confirmed directly: SIRT/SART/CGLS's `num_iter` is full sweeps
over every projection (their own algorithms' usual convention), but
ASTRA's plain 'ART' counts individual ray (row) updates. On a
300x300-pixel, 300-projection test case, num_iter=20/50 reconstructed
to exactly all zeros (nothing had happened yet); on a smaller
64x64-pixel, 240-projection case it wasn't literally zero but was
still barely started (correlation with ground truth ~0.35 vs 0.7+).
num_iter=n_projections*width (one full sweep-equivalent) gave a real
result in both cases. Pass something on that order for ART, not the
10s-100s that work fine for the other three -- there is no warning
from ASTRA when this is too low, it just silently looks like a broken
or badly under-converged result.

`EM_CUDA` (expectation-maximization) exists in astra_reconstruction
too but has no CPU counterpart at all, unlike these five, which all
have both a plain and a "_CUDA" form -- left out so every backend here
keeps working the same way regardless of device.

astra_reconstruction() falls back to the plain (CPU) method itself,
with its own warning, if "_CUDA" was requested but no GPU is available
-- the same numba.cuda check it uses internally (re-exported as
algotom.rec.reconstruction.cuda, same as fbp_reconstruction's) is what
actual_device() reports here too, for consistency with algotom_fbp.py.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from reconstruction.backend import Geometry, ReconstructionBackend, register_backend
from reconstruction.backends._algotom_common import algotom_reconstruction


@register_backend
class AlgotomBpBackend(ReconstructionBackend):
    name = "algotom_bp"
    supported_geometries = {"parallel"}
    requires = ("algotom", "astra")   # astra: the astra-toolbox package
    _method_base = "BP"

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
        method = f"{self._method_base}_CUDA" if gpu else self._method_base

        angles_rad = np.radians(angles)
        n_slices = sinograms.shape[1]
        centers = np.broadcast_to(np.asarray(center, dtype=float), (n_slices,))

        images = [
            rec.astra_reconstruction(sinograms[:, i, :], float(centers[i]), angles=angles_rad, method=method, **options)
            for i in range(n_slices)
        ]
        return np.stack(images)

    def actual_device(self, device: Optional[str] = None) -> str:
        wants_gpu = True if device is None else (device == "gpu")
        if not wants_gpu:
            return "cpu"
        return "gpu" if algotom_reconstruction(self.name).cuda.is_available() else "cpu"


@register_backend
class AlgotomSirtBackend(AlgotomBpBackend):
    name = "algotom_sirt"
    _method_base = "SIRT"
    is_iterative = True


@register_backend
class AlgotomSartBackend(AlgotomBpBackend):
    name = "algotom_sart"
    _method_base = "SART"
    is_iterative = True


@register_backend
class AlgotomArtBackend(AlgotomBpBackend):
    name = "algotom_art"
    _method_base = "ART"
    is_iterative = True


@register_backend
class AlgotomCglsBackend(AlgotomBpBackend):
    name = "algotom_cgls"
    _method_base = "CGLS"
    is_iterative = True
