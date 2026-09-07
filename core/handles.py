"""
DataHandle: a thin wrapper around either an in-memory numpy array or a
still-open HDF5 dataset, exposing the same interface (`.get()`,
`.shape`, `.storage_mode`) regardless of which.

ProjectObject stores these instead of raw arrays, so it stays cheap to
create and inspect even when the underlying data is large.

TIFF data is always loaded fully into RAM at import time, so
TiffBackend only ever produces RamHandle. Only HDF5 can produce the
disk-backed variant.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

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
        memory (h5py datasets expose .shape without reading data)."""

    @property
    @abstractmethod
    def storage_mode(self) -> str:
        """Either "ram" or "disk"."""


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


class Hdf5DiskHandle(DataHandle):
    """
    Wraps a live h5py.Dataset. `.get()` triggers a disk read at that
    moment; data is not loaded until requested.

    Requires the underlying h5py.File to stay open. ProjectObject keeps
    a reference to that open file alive for as long as any
    Hdf5DiskHandle from it exists (see ProjectObject._open_resource in
    project.py).
    """

    def __init__(self, dataset):
        self._dataset = dataset

    def get(self, index=None) -> np.ndarray:
        return self._dataset[()] if index is None else self._dataset[index]

    @property
    def shape(self) -> tuple:
        return self._dataset.shape

    @property
    def storage_mode(self) -> str:
        return "disk"

    def materialize(self) -> RamHandle:
        """
        Read the entire dataset into memory and return it as a
        RamHandle. Explicit, one-way conversion; see
        DataInterface.materialize_to_ram().
        """
        return RamHandle(self._dataset[()])
