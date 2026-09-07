"""
ProjectObject: represents one loaded dataset as it moves through the
pipeline.

ProjectObject holds data; it does not fetch or produce it. Two other
components do that work:
  - data_import/importer.py's DataImporter builds ProjectObjects.
  - core/data_interface.py's DataInterface reads from them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from core.metadata import MetadataBundle
from core.handles import DataHandle


@dataclass
class ProjectObject:
    """
    One tomography dataset as understood by the software at this point
    in the pipeline. Created by DataImporter, read and possibly updated
    by every later stage.

    Fields:
        format: "hdf5" or "tiff".
        storage_mode: "ram" or "disk". Always "ram" for tiff. For hdf5,
            depends on what was requested at import time, and can
            change later via DataInterface.materialize_to_ram().
        metadata: the four-tier metadata bundle; see core/metadata.py.
        data_handles: array data wrapped in DataHandle objects (see
            core/handles.py), keyed by name, e.g.
            data_handles["projections"], data_handles["flats"],
            data_handles["darks"].
        vendor_plugin_name: which plugin imported this file.
    """

    format: Literal["hdf5", "tiff"]
    storage_mode: Literal["ram", "disk"]
    metadata: MetadataBundle
    data_handles: dict[str, DataHandle] = field(default_factory=dict)
    vendor_plugin_name: Optional[str] = None

    # Free-form info about where the data came from (e.g. the PathMap a
    # plugin produced, or the original file path). A plain dict so
    # core/ does not need to import from data_import/.
    source_info: dict = field(default_factory=dict)

    # If storage_mode is "disk", the open h5py.File must stay alive for
    # the disk handles above to keep working. Stored here so
    # ProjectObject owns its own cleanup via close().
    _open_resource: Optional[object] = field(default=None, repr=False)

    def is_ready_for_import(self) -> bool:
        """True if there are no missing mandatory_hard fields."""
        return not self.metadata.missing_hard()

    def needs_user_input(self) -> bool:
        """True if there are mandatory_soft fields with no value yet."""
        return bool(self.metadata.missing_soft())

    def missing_hard_fields(self) -> list[str]:
        """See MetadataBundle.missing_hard()."""
        return self.metadata.missing_hard()

    def missing_soft_fields(self) -> list[str]:
        """See MetadataBundle.missing_soft()."""
        return self.metadata.missing_soft()

    def close(self):
        """
        Release any open file resource (e.g. an open h5py.File when
        storage_mode is "disk"). Safe to call even if there is nothing
        to close.
        """
        if self._open_resource is not None and hasattr(self._open_resource, "close"):
            self._open_resource.close()

    def __repr__(self):
        return (
            f"ProjectObject(format={self.format}, storage_mode={self.storage_mode}, "
            f"vendor={self.vendor_plugin_name}, handles={list(self.data_handles)})"
        )
