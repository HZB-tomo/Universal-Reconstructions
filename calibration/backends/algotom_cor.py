"""
Center-of-rotation finding via algotom.prep.calculation.find_center_vo --
the first, reference calibration backend. Needs the `algotom` package,
the same optional dependency preprocessing/corrections.py and
reconstruction/backends/algotom_fbp.py already use -- imported lazily,
with a plain ImportError telling you what to install if it's missing,
never a mandatory dependency of this project.

If algotom is ever unavailable or removed, this backend is NOT a black
box we'd be stuck without -- the algorithm it wraps is fully
reimplementable from its own published source:

    Vo, N.T., Drakopoulos, M., Atwood, R.C., Reinhard, C., "Reliable
    method for calculating the center of rotation in parallel-beam
    tomography," Optics Express 22(16), 19078-19086 (2014).
    https://doi.org/10.1364/OE.22.019078

A replacement backend only has to satisfy CalibrationBackend's one
method (find_center()); nothing in calibration/api.py or anything that
calls it would need to change.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from calibration.backend import CalibrationBackend, register_backend


def _algotom_calculation():
    try:
        import algotom.prep.calculation as calc
    except ImportError as e:
        raise ImportError(
            "The 'algotom' calibration backend requires the 'algotom' package, which isn't installed. "
            "Install it with: pip install algotom"
        ) from e
    return calc


@register_backend
class AlgotomCorBackend(CalibrationBackend):
    name = "algotom"

    def find_center(
        self,
        sino_180: np.ndarray,
        start: Optional[float] = None,
        stop: Optional[float] = None,
        **options,
    ) -> float:
        calc = _algotom_calculation()
        return float(calc.find_center_vo(np.asarray(sino_180), start=start, stop=stop, **options))
