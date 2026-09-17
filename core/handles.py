"""
DataHandle: a thin wrapper around either an in-memory numpy array, a
still-open HDF5 dataset, or a folder of TIFF files, exposing the same
interface (`.get()`, `.shape`, `.storage_mode`, `.materialize()`)
regardless of which.

ProjectObject stores these instead of raw arrays, so it stays cheap to
create and inspect even when the underlying data is large.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np


class DataHandle(ABC):
    """Common interface every handle implements."""

    @abstractmethod
    def get(self, index=None) -> np.ndarray:
        """
        Return the array data. If `index` is None, return the whole
        array; otherwise `index` is passed through to numpy-style
        indexing.
        """

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """The array's shape, without necessarily loading it into
        memory (h5py datasets and TIFF file headers both expose shape
        without reading pixel data)."""

    @property
    @abstractmethod
    def storage_mode(self) -> str:
        """Either "ram" or "disk"."""

    @abstractmethod
    def materialize(self) -> "RamHandle":
        """
        Return this data as a RamHandle: a no-op returning self if
        already RAM-backed, or a one-way disk-to-RAM read otherwise.
        See DataInterface.materialize_to_ram().
        """


class RamHandle(DataHandle):
    """Wraps a plain numpy array already in memory."""

    def __init__(self, array: np.ndarray):
        self._array = array

    def get(self, index=None) -> np.ndarray:
        return self._array if index is None else self._array[index]

    @property
    def shape(self) -> tuple:
        return self._array.shape

    @property
    def storage_mode(self) -> str:
        return "ram"

    def materialize(self) -> "RamHandle":
        return self


class Hdf5DiskHandle(DataHandle):
    """
    Wraps a live h5py.Dataset. `.get()` triggers a disk read at that
    moment; data is not loaded until requested.

    Requires the underlying h5py.File to stay open. ProjectObject keeps
    a reference to that open file alive for as long as any
    Hdf5DiskHandle from it exists (see ProjectObject._open_resource in
    project.py).

    `index_range`, when given, restricts this handle to that leading-
    axis slice of `dataset` -- e.g. a vendor whose flats are frames
    0-19 of the same dataset the projections come from (see
    data_import/vendors/bamline.py) points both PathMap roles at one
    dataset and gives each its own index_range. Handle-local index 0
    then means `dataset[index_range.start]`, not `dataset[0]`. Only a
    step of 1 (or None) is supported -- the one shape this is ever
    needed for is a contiguous frame range, not a strided one.
    """

    def __init__(self, dataset, index_range: Optional[slice] = None):
        self._dataset = dataset
        if index_range is not None and index_range.step not in (None, 1):
            raise ValueError(f"Hdf5DiskHandle only supports a step of 1, got {index_range.step!r}")
        self._start, self._stop, _ = (
            (0, dataset.shape[0], 1) if index_range is None else index_range.indices(dataset.shape[0])
        )

    def _translate(self, leading):
        if isinstance(leading, slice):
            offset_start, offset_stop, offset_step = leading.indices(self._stop - self._start)
            return slice(self._start + offset_start, self._start + offset_stop, offset_step)
        if leading is None:
            return slice(self._start, self._stop)
        return self._start + leading

    def get(self, index=None) -> np.ndarray:
        if index is None:
            return self._dataset[self._start:self._stop]
        leading, rest = (index[0], index[1:]) if isinstance(index, tuple) else (index, ())
        translated = self._translate(leading)
        return self._dataset[(translated, *rest)] if rest else self._dataset[translated]

    @property
    def shape(self) -> tuple:
        return (self._stop - self._start, *self._dataset.shape[1:])

    @property
    def storage_mode(self) -> str:
        return "disk"

    def materialize(self) -> RamHandle:
        return RamHandle(self._dataset[self._start:self._stop])


class TiffDiskHandle(DataHandle):
    """
    Wraps a sorted list of per-slice TIFF file paths. Nothing is read
    until .get() or .shape is actually called; .shape reads only the
    first file's header, not its pixel data.
    """

    def __init__(self, files: list[Path]):
        self._files = files
        self._shape = None  # cached after first probe

    def get(self, index=None) -> np.ndarray:
        """
        `index` may address just the leading (file) axis, or be a
        tuple whose first element addresses the file axis and whose
        remaining elements index into each frame after it's read (e.g.
        `(slice(None), rows, slice(None))`, the shape
        preprocessing.normalize_rows() indexes with) -- mirroring what
        h5py.Dataset.__getitem__ already does natively for
        Hdf5DiskHandle, which a plain list of file paths doesn't get
        for free.
        """
        import tifffile  # imported here so this package stays importable without tifffile installed

        if index is None:
            return np.stack([tifffile.imread(f) for f in self._files], axis=0)

        file_index, frame_index = (index[0], index[1:]) if isinstance(index, tuple) else (index, ())

        if isinstance(file_index, int):
            frame = tifffile.imread(self._files[file_index])
            return frame[frame_index] if frame_index else frame

        stack = np.stack([tifffile.imread(f) for f in self._files[file_index]], axis=0)
        return stack[(slice(None), *frame_index)] if frame_index else stack

    @property
    def shape(self) -> tuple:
        if self._shape is None:
            import tifffile

            with tifffile.TiffFile(self._files[0]) as tf:
                page_shape = tf.pages[0].shape
            self._shape = (len(self._files), *page_shape)
        return self._shape

    @property
    def storage_mode(self) -> str:
        return "disk"

    def materialize(self) -> RamHandle:
        return RamHandle(self.get(None))
