"""
calibration/: finds center-of-rotation and rotation-axis tilt from
normalized sinograms/projections -- see doc/CALIBRATION.md.
"""
from calibration import backends  # noqa: F401  (registers built-in backends on import)
from calibration.api import TiltAndCenterResult, find_center, find_tilt_and_center
from calibration.backend import CalibrationBackend, get_backend, register_backend

__all__ = [
    "find_center",
    "find_tilt_and_center",
    "TiltAndCenterResult",
    "CalibrationBackend",
    "get_backend",
    "register_backend",
]
