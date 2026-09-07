"""
Registers each calibration backend module by importing it.

To add a backend: create a new module here, define a class inheriting
from CalibrationBackend, decorate it with @register_backend, and add an
import line below.
"""
from calibration.backends import algotom_cor  # noqa: F401
