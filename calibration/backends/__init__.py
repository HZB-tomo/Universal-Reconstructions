"""
Every module in this folder is one calibration backend. Importing THIS
package is what actually registers them -- each module uses the
@register_backend decorator from calibration/backend.py, which only
runs when the module is imported. Same pattern as
data_import/vendors/__init__.py and reconstruction/backends/__init__.py.

To add support for another library (or a from-scratch reimplementation
of algotom_cor.py's algorithm, if algotom is ever unavailable): create
a new file here, define a class inheriting from CalibrationBackend,
decorate it with @register_backend, and add an import line below.
"""
from calibration.backends import algotom_cor  # noqa: F401
