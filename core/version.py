"""
The version of this software, in one place. Both pyproject.toml files read
it (dynamic version), the CLI prints it (`tomo --version`) and the HDF5 files
the software writes record it (/entry/software, see core/urct_format.py).

Semantic versioning: MAJOR.MINOR.PATCH; before 1.0 a new minor version may
change behavior or file layouts. Change it here, add a section to
CHANGELOG.md, and tag the commit `vX.Y.Z`.
"""
__version__ = "0.2.0"
