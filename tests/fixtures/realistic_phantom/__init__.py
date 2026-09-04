"""
A standalone, realistic CT test-data generator: a Shepp-Logan phantom
projected through a simulated detector with controllable center-of-
rotation deviation, detector tilt, flat/dark-field imperfections, and
noise -- everything needed to exercise normalization, ring removal,
center-of-rotation finding, and tilt correction against data with a
known ground truth, not just a geometrically perfect toy volume.

Independent of the rest of the project: only numpy/scipy for the
simulation itself, h5py only inside write_hdf5() (imported lazily, the
same pattern used everywhere else in this codebase). The one place this
does reach into the main package is to_project_object(), which builds a
core.project.ProjectObject directly -- that's an explicit, optional
convenience, not a hard dependency of the simulation itself.

See doc/REALISTIC_PHANTOM.md for the physical model and full parameter
reference, and data_import/vendors/generic_parallel_synchrotron.py for
the vendor plugin this generator's files are meant to be read back
through.

Public API:
    generate(...)              -> PhantomProject
    write_hdf5(phantom, path)  -> None
    to_project_object(phantom) -> core.project.ProjectObject
"""
from tests.fixtures.realistic_phantom.generate import PhantomProject, generate, to_project_object, write_hdf5

__all__ = ["PhantomProject", "generate", "write_hdf5", "to_project_object"]
