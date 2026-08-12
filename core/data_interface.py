"""
This file defines DataInterface: the ONLY class the rest of the software
(reconstruction, viewing, GUI, ...) should use to actually get array data
or metadata values out of a ProjectObject.

Why not just read straight from ProjectObject.data_handles? Because that
would leak the RAM-vs-disk distinction into every module that needs data.
DataInterface exists purely to hide that distinction: whether you call
`get_projections()` on a RAM-backed or disk-backed project, you get a
numpy array back either way, and you don't need to know which one it was.

If we later add a third storage mode (e.g. a memory-mapped or networked
array library), only DataInterface and core/handles.py need to change --
every module that already uses DataInterface keeps working unmodified.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from core.project import ProjectObject


class DataInterface:
    """
    Wraps a single ProjectObject and exposes convenient, storage-agnostic
    methods for getting data out of it. Create one per project you're
    working with: `di = DataInterface(my_project)`.
    """

    def __init__(self, project: ProjectObject):
        self.project = project

    def _handle(self, key: str):
        """Internal helper: fetch a handle by name, with a clear error
        if it doesn't exist, instead of a confusing KeyError from deep
        inside a dict lookup."""
        if key not in self.project.data_handles:
            raise KeyError(f"No data handle registered for '{key}'")
        return self.project.data_handles[key]

    def get_projections(self, index=None) -> np.ndarray:
        """Return the projection images. `index` can be used to fetch
        only a subset (e.g. index=0 for just the first angle) without
        necessarily loading everything -- see DataHandle.get()."""
        return self._handle("projections").get(index)

    def get_flats(self, index=None) -> Optional[np.ndarray]:
        """Return the flat-field images, or None if this project has
        none (some data may already be flat-field corrected upstream,
        in which case there's nothing to fetch here)."""
        if "flats" not in self.project.data_handles:
            return None
        return self._handle("flats").get(index)

    def get_darks(self, index=None) -> Optional[np.ndarray]:
        """Same idea as get_flats(), for dark-field images."""
        if "darks" not in self.project.data_handles:
            return None
        return self._handle("darks").get(index)

    def shape_of(self, key: str) -> tuple:
        """Get an array's shape (e.g. shape_of("projections")) without
        loading the actual data -- useful for sizing UI elements or
        planning memory usage before committing to a full load."""
        return self._handle(key).shape

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Read a metadata value (e.g. get_metadata("voxel_size")).
        Just forwards to the project's MetadataBundle -- this method
        exists so calling code only ever needs to import DataInterface,
        not also reach into project.metadata directly."""
        return self.project.metadata.get(key, default)

    def set_computed(self, key: str, value: Any):
        """Record a freshly computed metadata value (e.g. after running
        a center-of-rotation estimation algorithm). Forwards to
        MetadataBundle.set_computed() -- see core/metadata.py for why
        this overwrites rather than keeping history."""
        self.project.metadata.set_computed(key, value)

    def materialize_to_ram(self):
        """
        Convert every disk-backed handle in this project into a RAM
        handle, and update the project's storage_mode accordingly. This
        is how a user says "actually, load everything into memory now"
        after starting in disk mode. One-directional: there's no
        automatic "demote back to disk" the other way, since that would
        require deciding where on disk to write it, which isn't this
        method's job.
        """
        from core.handles import Hdf5DiskHandle  # local import avoids a
        # module-level dependency cycle risk if handles.py ever needs
        # something from here in the future.

        for key, handle in list(self.project.data_handles.items()):
            if isinstance(handle, Hdf5DiskHandle):
                self.project.data_handles[key] = handle.materialize()
        self.project.storage_mode = "ram"
