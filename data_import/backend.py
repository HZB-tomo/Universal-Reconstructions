"""
This file answers a completely different question from plugin.py: not
"where is the data" (vendor-specific, a plugin's job) but "how do I
physically read bytes given a location" (format-specific, this file's
job). A FormatBackend never needs to know which vendor wrote a file --
it just knows how HDF5 or TIFF work in general.

Why keep this separate from plugins: the SAME backend is reused by every
plugin of that format. Hdf5Backend doesn't care whether the path it's
given came from a Zeiss plugin or a Bruker plugin -- it just opens that
path. This also means when we later add EXPORTING data (e.g. writing a
reconstructed volume back to hdf5), we reuse Hdf5Backend for that too,
rather than writing separate read/write code per vendor.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np

from core.handles import DataHandle, RamHandle, Hdf5DiskHandle


class FormatBackend(ABC):
    """
    Common interface both backends below implement. DataImporter only
    ever talks to backends through these three methods -- it never
    contains any hdf5-specific or tiff-specific code itself.
    """

    format: Literal["hdf5", "tiff"]

    @abstractmethod
    def open(self, path):
        """
        Physically open the container and return a handle for further
        use. This returned handle is what gets passed to a plugin's
        matches() / get_path_map() / parse_metadata(), and later to
        build_data_handle() below. For HDF5 this is an open h5py.File.
        For TIFF this is just the folder as a Path (nothing to "open"
        up front -- individual files get read lazily as needed).
        """

    @abstractmethod
    def build_data_handle(self, opened, internal_path: str, storage_mode: str) -> DataHandle:
        """
        Given the opened container and one path from a PathMap (e.g.
        path_map.projections), actually read the data and return it
        wrapped in a DataHandle. `storage_mode` is "ram" or "disk",
        exactly as the user requested when calling DataImporter -- this
        is the one place that choice actually takes effect.
        """

    @abstractmethod
    def close(self, opened):
        """Release any resources opened by open() (e.g. close the
        h5py.File). Safe to call even if there's nothing to release."""


class Hdf5Backend(FormatBackend):
    """
    Knows how to open HDF5 files and read datasets from them, in either
    RAM or disk (lazy) mode. Has zero knowledge of which vendor wrote
    any given file -- it just operates on whatever internal path it's
    told to read.
    """

    format = "hdf5"

    def open(self, path):
        import h5py  # imported here (not at module level) so the rest
        # of this package stays importable even in an environment that
        # only has tifffile installed and not h5py, e.g. if someone is
        # only working with tiff data and hasn't installed h5py yet.

        # Always open read-only for now -- writing back to an existing
        # file, or exporting a NEW hdf5 file, is a separate concern for
        # later (reconstruction/export modules), not something the
        # import path needs to handle.
        return h5py.File(path, "r")

    def build_data_handle(self, opened, internal_path: str, storage_mode: str) -> DataHandle:
        dataset = opened[internal_path]
        if storage_mode == "ram":
            # dataset[()] reads the ENTIRE dataset from disk right now
            # and returns it as a plain numpy array.
            return RamHandle(dataset[()])
        # storage_mode == "disk": wrap the still-open dataset without
        # reading anything yet. Actual reads happen later, on demand,
        # via Hdf5DiskHandle.get() (see core/handles.py).
        return Hdf5DiskHandle(dataset)

    def close(self, opened):
        opened.close()


class TiffBackend(FormatBackend):
    """
    Knows how to read a folder of TIFF files into a single stacked numpy
    array. Per our decision, TIFF data is ALWAYS fully loaded into RAM --
    there is no lazy/disk-backed mode for tiff, so build_data_handle()
    below always returns a RamHandle regardless of what storage_mode is
    requested (DataImporter also enforces this earlier, by rejecting
    storage_mode="disk" for tiff projects before we ever get here).
    """

    format = "tiff"

    def open(self, path):
        # Nothing to actually "open" for a folder -- just normalize to
        # a Path object for the rest of the code to use consistently.
        return Path(path)

    def build_data_handle(self, opened: Path, internal_path: str, storage_mode: str) -> DataHandle:
        import tifffile  # imported here for the same reason h5py is
        # imported lazily in Hdf5Backend -- keeps this package usable
        # even without every optional dependency installed.

        # internal_path here is a glob pattern relative to the opened
        # folder, e.g. "projections/*.tif" -- see PathMap for where this
        # comes from.
        files = sorted(opened.glob(internal_path))
        if not files:
            raise FileNotFoundError(f"No files matched pattern '{internal_path}' under {opened}")
        stack = np.stack([tifffile.imread(f) for f in files], axis=0)
        return RamHandle(stack)

    def close(self, opened):
        pass  # a folder isn't a file handle, nothing to release
