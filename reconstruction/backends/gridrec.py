"""
Gridrec reconstruction backend, wrapping the separate `urct_gridrec` package
(gridrec_project/ in this repository): Fourier-gridding filtered backprojection.

Install it with the `gridrec` extra (see the README). It is a pure-Python
package (Numba for the speed), so it works with a plain `pip install` -- unlike
TomoPy's gridrec behind `algotom_gridrec`, which needs conda.

Why use it: cost grows with width^2 x log(width) instead of width^2 x angles,
so it is several times to tens of times faster than FBP on the CPU, with the
same accuracy and the same output scale (see gridrec_project/README.md).
It is also deterministic: the same input always gives the same bits.

Devices: CPU is the default and the tested path. `device="gpu"` runs the
FFTs on CuPy and the gridding as a numba.cuda kernel; that path is verified
against the CPU on numba's CUDA simulator but has not been timed or run on a
real GPU yet, so it is only used when asked for explicitly. "Auto" means CPU.

The input is transmission (what normalization produces); the -log is taken
inside, like the algotom backends do.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from reconstruction.backend import Geometry, ReconstructionBackend, register_backend

_INSTALL_HINT = (
    "The 'gridrec' reconstruction backend needs the 'urct_gridrec' package. "
    "Install it with: pip install -e ./gridrec_project   (or, once published: pip install tomo_project[gridrec])"
)


def _gridrec():
    try:
        import urct_gridrec
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e
    return urct_gridrec


@register_backend
class GridrecBackend(ReconstructionBackend):
    name = "gridrec"
    supported_geometries = {"parallel"}
    requires = ("urct_gridrec", "numba")

    def reconstruct(
        self,
        sinograms: np.ndarray,
        angles: np.ndarray,
        center,
        geometry: Geometry,
        device: Optional[str] = None,
        **options,
    ) -> np.ndarray:
        gr = _gridrec()
        return gr.gridrec(sinograms, angles, center, device=self.actual_device(device), **options)

    def actual_device(self, device: Optional[str] = None) -> str:
        # Only an explicit "gpu" request uses the GPU path (see the module docstring).
        if device != "gpu":
            return "cpu"
        try:
            return "gpu" if _gridrec().gpu_available() else "cpu"
        except ImportError:
            return "cpu"


def warm_up() -> None:
    """Compile (or load from Numba's cache) the CPU kernels with a tiny reconstruction, so the
    first real preview does not pay the compile time. Safe to call from a background thread;
    does nothing if gridrec is not installed."""
    if not GridrecBackend().is_available():
        return
    gr = _gridrec()
    rng = np.random.default_rng(0)
    sino = rng.uniform(0.2, 1.0, (24, 2, 32)).astype(np.float32)
    gr.gridrec(sino, np.linspace(0, 180, 24, endpoint=False), 15.5, device="cpu")
