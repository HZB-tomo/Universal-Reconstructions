"""
FormatBackend: how to physically read hdf5/tiff data, independent of
vendor. A FormatBackend does not know which vendor wrote a file; it
only knows how the format itself works. The same backend is reused by
every plugin of that format.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np

from core.handles import DataHandle, RamHandle, Hdf5DiskHandle


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
    def build_data_handle(self, opened, internal_path: str, storage_mode: str) -> DataHandle:
        """
        Read the data at `internal_path` (from a PathMap, e.g.
        path_map.projections) and return it wrapped in a DataHandle.
        `storage_mode` is "ram" or "disk".
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

    def build_data_handle(self, opened, internal_path: str, storage_mode: str) -> DataHandle:
        dataset = opened[internal_path]
        if storage_mode == "ram":
            return RamHandle(dataset[()])
        return Hdf5DiskHandle(dataset)

    def close(self, opened):
        opened.close()


class TiffBackend(FormatBackend):
    """Reads a folder of TIFF files into a stacked numpy array. TIFF
    data is always fully loaded into RAM; build_data_handle() always
    returns a RamHandle."""

    format = "tiff"

    def open(self, path):
        return Path(path)

    def build_data_handle(self, opened: Path, internal_path: str, storage_mode: str) -> DataHandle:
        import tifffile  # imported here so this package stays importable without tifffile installed

        # internal_path is a glob pattern relative to `opened`, e.g. "projections/*.tif"
        files = sorted(opened.glob(internal_path))
        if not files:
            raise FileNotFoundError(f"No files matched pattern '{internal_path}' under {opened}")
        stack = np.stack([tifffile.imread(f) for f in files], axis=0)
        return RamHandle(stack)

    def close(self, opened):
        pass
