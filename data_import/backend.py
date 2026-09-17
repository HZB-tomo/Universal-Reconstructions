"""
FormatBackend: how to physically read hdf5/tiff data, independent of
vendor. A FormatBackend does not know which vendor wrote a file; it
only knows how the format itself works. The same backend is reused by
every plugin of that format.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from core.handles import DataHandle, RamHandle, Hdf5DiskHandle, TiffDiskHandle


class FormatBackend(ABC):
    """Common interface both backends implement."""

    format: Literal["hdf5", "tiff"]

    @abstractmethod
    def open(self, path):
        """
        Open the container and return a handle. Passed to a plugin's
        matches() / get_path_map() / parse_metadata(), and to
        build_data_handle() below. For HDF5 this is an open h5py.File.
        For TIFF this is the folder as a Path.
        """

    @abstractmethod
    def build_data_handle(self, opened, internal_path: str, storage_mode: str, index_range: Optional[slice] = None) -> DataHandle:
        """
        Read the data at `internal_path` (from a PathMap, e.g.
        path_map.projections) and return it wrapped in a DataHandle.
        `storage_mode` is "ram" or "disk".

        `index_range`, when given (from PathMap's matching *_range
        field, e.g. projections_range), restricts the handle to that
        leading-axis slice of the dataset/file-list at `internal_path`
        -- for a vendor where multiple roles share one physical dataset
        rather than each having their own (see vendors/bamline.py).
        None (the default) means the whole thing, exactly as before
        this parameter existed.
        """

    @abstractmethod
    def close(self, opened):
        """Release resources opened by open(). Safe to call even if
        there is nothing to release."""


class Hdf5Backend(FormatBackend):
    """Opens HDF5 files and reads datasets from them, in RAM or disk
    (lazy) mode. Has no knowledge of which vendor wrote a file."""

    format = "hdf5"

    def open(self, path):
        import h5py  # imported here so this package stays importable without h5py installed

        return h5py.File(path, "r")

    def build_data_handle(self, opened, internal_path: str, storage_mode: str, index_range: Optional[slice] = None) -> DataHandle:
        dataset = opened[internal_path]
        if storage_mode == "ram":
            return RamHandle(dataset[()] if index_range is None else dataset[index_range])
        return Hdf5DiskHandle(dataset, index_range=index_range)

    def close(self, opened):
        opened.close()


class TiffBackend(FormatBackend):
    """Reads a folder of TIFF files as a stack, in RAM or disk (lazy)
    mode. Has no knowledge of which vendor wrote a file."""

    format = "tiff"

    def open(self, path):
        return Path(path)

    def build_data_handle(self, opened: Path, internal_path: str, storage_mode: str, index_range: Optional[slice] = None) -> DataHandle:
        # internal_path is a glob pattern relative to `opened`, e.g. "projections/*.tif"
        files = sorted(opened.glob(internal_path))
        if not files:
            raise FileNotFoundError(f"No files matched pattern '{internal_path}' under {opened}")
        if index_range is not None:
            files = files[index_range]

        if storage_mode == "disk":
            return TiffDiskHandle(files)

        import tifffile  # imported here so this package stays importable without tifffile installed

        stack = np.stack([tifffile.imread(f) for f in files], axis=0)
        return RamHandle(stack)

    def close(self, opened):
        pass
