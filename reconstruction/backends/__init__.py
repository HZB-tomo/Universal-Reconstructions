"""
Registers each reconstruction backend module by importing it.

To add a backend: create a new module here, define a class inheriting
from ReconstructionBackend, decorate it with @register_backend, and add
an import line below.
"""
from reconstruction.backends import algotom_fbp  # noqa: F401
from reconstruction.backends import gridrec  # noqa: F401
from reconstruction.backends import algotom_gridrec  # noqa: F401
from reconstruction.backends import algotom_bpf  # noqa: F401
from reconstruction.backends import algotom_dfi  # noqa: F401
from reconstruction.backends import algotom_astra  # noqa: F401
