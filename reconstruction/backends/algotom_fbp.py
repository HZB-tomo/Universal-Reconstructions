"""
Parallel-beam FBP via algotom.rec.reconstruction.fbp_reconstruction --
the first, reference reconstruction backend. Needs the `algotom`
package, the same optional dependency preprocessing/corrections.py
already uses -- imported lazily, with a plain ImportError telling you
what to install if it's missing, never a mandatory dependency of this
project.

algotom's fbp_reconstruction() takes exactly one sinogram at a time --
see reconstruction/backend.py's ReconstructionBackend.reconstruct()
docstring for why the contract still only asks for one method. This
implementation loops over the requested stack, one call per sinogram.
That's deliberately NOT parallelized (no joblib, no GPU batching) --
it's the simplest correct reference implementation, not a claim that
it's the fastest one possible. A future revision of this backend, or a
different one entirely (raw ASTRA, tomopy, a GPU-batch-native library),
is exactly where genuine batch/parallel performance work belongs --
nothing about the contract needs to change for that, only this method's
body.
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

        # algotom's own default is gpu=True; device=None means "use
        # this backend's default" per the contract, so only override
        # when the caller actually asked for one explicitly.
        gpu = True if device is None else (device == "gpu")

        angles_rad = np.radians(angles)
        n_slices = sinograms.shape[1]  # sinograms is (n_angles, n_slices, n_columns) -- see backend.py
        centers = np.broadcast_to(np.asarray(center, dtype=float), (n_slices,))

        images = [
            rec.fbp_reconstruction(sinograms[:, i, :], float(centers[i]), angles=angles_rad, gpu=gpu, **options)
            for i in range(n_slices)
        ]
        # "per pixel" attenuation values, float32 -- api.reconstruct() handles
        # unit conversion (1/detector_pixel_size) and the final dtype
        # guarantee centrally, so this backend doesn't need to repeat either.
        return np.stack(images)
