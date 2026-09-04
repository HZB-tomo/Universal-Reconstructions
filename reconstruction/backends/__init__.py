"""
Every module in this folder is one reconstruction backend. Importing
THIS package registers them -- each module uses the @register_backend
decorator from reconstruction/backend.py, which only runs when the
module is imported. Same pattern as data_import/vendors/__init__.py.

To add a new backend: create a new file here, define a class
inheriting from ReconstructionBackend, decorate it with
@register_backend, and add an import line for it below.
"""
from reconstruction.backends import algotom_fbp  # noqa: F401
