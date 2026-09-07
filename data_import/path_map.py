"""
PathMap: a plain data holder for where things are in one file/folder,
as determined by a vendor plugin. Both plugin.py (produces PathMap
objects) and backend.py (consumes them) import this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PathMap:
    """
    Points to where the key pieces of data live inside one file/folder.
    For HDF5, these are internal dataset paths (e.g. "/exchange/data").
    For TIFF, these are glob patterns relative to the project folder
    (e.g. "projections/*.tif").

    A vendor plugin's get_path_map() method returns one of these; the
    format backend then uses it to actually read the arrays (see
    backend.py's FormatBackend.build_data_handle()).
    """

    projections: Optional[str] = None
    flats: Optional[str] = None
    darks: Optional[str] = None
    angles: Optional[str] = None
    # Where the plugin found metadata (e.g. a sidecar "metadata.json"
    # path, or an hdf5 group).
    metadata_source: Optional[str] = None
    # Vendor-specific extras that do not fit the fields above.
    extra: dict = field(default_factory=dict)
