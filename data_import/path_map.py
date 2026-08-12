"""
Defines PathMap: a small, plain data holder for "where things are" in
one specific file/folder, as figured out by a vendor plugin.

Why this is its own tiny file rather than folded into plugin.py: both
plugin.py (which produces PathMap objects) and backend.py (which
consumes them to actually build data handles) need to import it, and
keeping it separate avoids plugin.py and backend.py needing to import
from each other.
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
    # path, or an hdf5 group). Not always used -- some vendors keep
    # metadata as hdf5 attributes rather than a separate location.
    metadata_source: Optional[str] = None
    # Escape hatch for anything vendor-specific that doesn't fit the
    # fields above (e.g. a second sidecar file, a calibration path).
    # Deliberately a free-form dict rather than trying to anticipate
    # every possible vendor quirk up front.
    extra: dict = field(default_factory=dict)
