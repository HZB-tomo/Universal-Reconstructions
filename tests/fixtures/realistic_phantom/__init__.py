"""
A CT test-data generator: a Shepp-Logan phantom projected through a
simulated detector with controllable center-of-rotation deviation,
rotation-axis tilt, flat/dark-field imperfections, and noise, with a
known ground truth.

Depends only on numpy/scipy, plus h5py inside write_hdf5(). Files
written by write_hdf5() are read back through
data_import/vendors/generic_parallel_synchrotron.py.

See doc/REALISTIC_PHANTOM.md for the physical model and parameter
reference.

Public API:
    generate(...)              -> PhantomProject
    write_hdf5(phantom, path)  -> None
    to_project_object(phantom) -> core.project.ProjectObject
"""
from tests.fixtures.realistic_phantom.generate import PhantomProject, generate, to_project_object, write_hdf5

__all__ = ["PhantomProject", "generate", "write_hdf5", "to_project_object"]
