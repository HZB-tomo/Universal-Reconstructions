"""
Finds center-of-rotation and rotation-axis tilt from normalized
sinograms/projections. See doc/CALIBRATION.md.
"""
from calibration import backends  # noqa: F401
from calibration.api import (
    CenterCandidate, CenterSearch, TiltResult, centers_for_rows, find_center, find_tilt, half_scan_slice,
    search_center, search_window,
)
from calibration.backend import CalibrationBackend, get_backend, register_backend

__all__ = [
    "find_center",
    "search_center",
    "search_window",
    "CenterSearch",
    "CenterCandidate",
    "find_tilt",
    "half_scan_slice",
    "centers_for_rows",
    "TiltResult",
    "CalibrationBackend",
    "get_backend",
    "register_backend",
]
