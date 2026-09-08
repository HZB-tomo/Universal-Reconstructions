"""
DataInterface: the class the rest of the software uses to read array
data or metadata values out of a ProjectObject, without needing to know
whether the project is RAM-backed or disk-backed.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from core.project import ProjectObject


class DataInterface:
    """
    Wraps a ProjectObject and exposes storage-agnostic methods for
    reading data from it: `di = DataInterface(my_project)`.
    """

    def __init__(self, project: ProjectObject):
        self.project = project

    def _handle(self, key: str):
        if key not in self.project.data_handles:
            raise KeyError(f"No data handle registered for '{key}'")
        return self.project.data_handles[key]

    def get_projections(self, index=None) -> np.ndarray:
        """Return the projection images. `index` fetches a subset (e.g.
        index=0 for the first angle) without loading everything."""
        return self._handle("projections").get(index)

    def get_flats(self, index=None) -> Optional[np.ndarray]:
        """Return the flat-field images, or None if this project has
        none."""
        if "flats" not in self.project.data_handles:
            return None
        return self._handle("flats").get(index)

    def get_darks(self, index=None) -> Optional[np.ndarray]:
        """Return the dark-field images, or None if this project has
        none."""
        if "darks" not in self.project.data_handles:
            return None
        return self._handle("darks").get(index)

    def shape_of(self, key: str) -> tuple:
        """An array's shape (e.g. shape_of("projections")) without
        loading the data."""
        return self._handle(key).shape

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Read a metadata value (e.g. get_metadata("voxel_size"))."""
        return self.project.metadata.get(key, default)

    def set_computed(self, key: str, value: Any):
        """Record a freshly computed metadata value. Forwards to
        MetadataBundle.set_computed()."""
        self.project.metadata.set_computed(key, value)

    def materialize_to_ram(self):
        """
        Convert every disk-backed handle in this project to a RAM
        handle, and update storage_mode accordingly. One-directional.
        Handles already in RAM are left as-is (DataHandle.materialize()
        is a no-op for RamHandle).
        """
        for key, handle in list(self.project.data_handles.items()):
            self.project.data_handles[key] = handle.materialize()
        self.project.storage_mode = "ram"
