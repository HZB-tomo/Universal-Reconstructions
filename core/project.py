"""
This file defines ProjectObject: the single object that represents "one
loaded dataset" and gets passed from module to module as the software
does its work (import -> reconstruction -> viewing -> ... in the future).

ProjectObject is deliberately kept "dumb" -- it holds data, it doesn't
know how to fetch/produce it. Two other things handle the actual work:
  - data_import/importer.py's DataImporter builds ProjectObjects
  - core/data_interface.py's DataInterface reads from them

Why keep ProjectObject this passive? Because every future module
(reconstruction, viewing, GUI) will need to read from and write to a
ProjectObject, and if ProjectObject itself contained loading/saving
logic, every one of those modules would end up depending on import-
specific or storage-specific code they don't actually need. Keeping it
a plain container means any module can hold a ProjectObject and only
import the pieces (DataInterface, etc.) that it actually uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from core.metadata import MetadataBundle
from core.handles import DataHandle


@dataclass
class ProjectObject:
    """
    Represents one tomography dataset as it's understood by the software
    at this point in the pipeline. Created by DataImporter, then read
    and possibly updated by every later stage.

    Fields:
        format: which container format the data came from. Right now
            always "hdf5" or "tiff", matching the two formats we support.
        storage_mode: whether the data currently lives in RAM or is
            being read lazily from disk. For tiff this is always "ram"
            (see data_import/backend.py). For hdf5 it depends on what
            the user asked for at import time, and can change later if
            DataInterface.materialize_to_ram() is called.
        metadata: the four-tier metadata bundle (voxel size, angles,
            center of rotation, etc.) -- see core/metadata.py.
        data_handles: the actual array data, but wrapped in DataHandle
            objects rather than raw numpy arrays (see core/handles.py),
            keyed by name -- e.g. data_handles["projections"],
            data_handles["flats"], data_handles["darks"].
        vendor_plugin_name: which plugin was used to import this file,
            kept mainly for debugging/logging ("why did it load this way?").
    """

    format: Literal["hdf5", "tiff"]
    storage_mode: Literal["ram", "disk"]
    metadata: MetadataBundle
    data_handles: dict[str, DataHandle] = field(default_factory=dict)
    vendor_plugin_name: Optional[str] = None

    # Free-form info about where the data came from (e.g. the PathMap a
    # plugin produced, or the original file path). Kept as a plain dict
    # rather than a typed object so core/ doesn't need to import
    # anything from data_import/ -- core stays independent of the
    # import machinery, and only data_import writes into this field.
    source_info: dict = field(default_factory=dict)

    # If storage_mode is "disk", something (the open h5py.File) needs to
    # stay alive for the disk handles above to keep working. We store
    # that "something" here so ProjectObject itself is responsible for
    # its own cleanup via close(), rather than making every caller
    # remember to keep a separate reference alive. Leading underscore
    # signals "internal, don't touch this directly from outside".
    _open_resource: Optional[object] = field(default=None, repr=False)

    def is_ready_for_import(self) -> bool:
        """
        True if there are no missing mandatory_hard fields. In practice
        DataImporter already refuses to hand back a ProjectObject that
        fails this check (it raises an error instead) -- this method
        exists so later code can still ask the question defensively,
        e.g. if a ProjectObject gets passed around and modified.
        """
        return not self.metadata.missing_hard()

    def needs_user_input(self) -> bool:
        """
        True if there are mandatory_soft fields with no value yet (e.g.
        voxel size wasn't in the file). A GUI or CLI should check this
        after import and prompt the user before letting them proceed to
        stages that need those fields.
        """
        return bool(self.metadata.missing_soft())

    def missing_hard_fields(self) -> list[str]:
        """Convenience passthrough -- see MetadataBundle.missing_hard()."""
        return self.metadata.missing_hard()

    def missing_soft_fields(self) -> list[str]:
        """Convenience passthrough -- see MetadataBundle.missing_soft()."""
        return self.metadata.missing_soft()

    def close(self):
        """
        Release any open file resource (e.g. an open h5py.File when
        storage_mode is "disk"). Call this when you're done with a
        disk-backed project, similar to closing a file handle. Safe to
        call even if there's nothing to close.
        """
        if self._open_resource is not None and hasattr(self._open_resource, "close"):
            self._open_resource.close()

    def __repr__(self):
        return (
            f"ProjectObject(format={self.format}, storage_mode={self.storage_mode}, "
            f"vendor={self.vendor_plugin_name}, handles={list(self.data_handles)})"
        )
