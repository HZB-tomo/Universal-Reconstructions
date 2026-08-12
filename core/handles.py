"""
This file solves one specific problem: the rest of the software wants to
say "give me the projections array" without caring whether that array is
sitting in RAM as a plain numpy array, or still living on disk inside an
open HDF5 file and only read when actually needed.

We solve it with a small class hierarchy called "handles". A handle is a
thin wrapper around either kind of storage that exposes the SAME interface
(`.get()`, `.shape`, `.storage_mode`) regardless of what's underneath.

Why this matters for the project structure: ProjectObject (see project.py)
stores these handles instead of raw arrays. That keeps ProjectObject
cheap to create, inspect, and eventually serialize (e.g. dump as JSON for
reproducibility) even when the underlying data is tens of gigabytes --
because ProjectObject itself never holds the actual array, just a
reference to where it lives.

Note: per our decision, TIFF data is ALWAYS loaded fully into RAM at
import time. So TiffBackend only ever produces RamHandle. Only HDF5 can
produce the disk-backed variant, because only HDF5 supports leaving data
on disk and reading slices from it lazily via h5py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DataHandle(ABC):
    """
    Common interface every handle must implement. Nothing outside this
    file should ever check "is this a RamHandle or a disk handle?" --
    if calling code needs to branch on that, it's a sign the interface
    here is missing something, and we should add a method instead.
    """

    @abstractmethod
    def get(self, index=None) -> np.ndarray:
        """
        Return the array data. If `index` is None, return the whole
        array. Otherwise `index` is passed straight through to numpy-style
        indexing (a slice, an integer, a tuple of slices, etc.), so
        callers can request e.g. just one projection image instead of
        the whole stack.
        """

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """The array's shape, without necessarily loading the whole
        array into memory to find out (h5py datasets expose .shape
        cheaply, without reading any data)."""

    @property
    @abstractmethod
    def storage_mode(self) -> str:
        """Either "ram" or "disk" -- mainly useful for logging/debugging
        and for ProjectObject.storage_mode to stay in sync with what its
        handles are actually doing."""


class RamHandle(DataHandle):
    """
    The simple case: the array is already a plain numpy array sitting in
    memory. This is what TIFF data always becomes (per our decision),
    and what HDF5 data becomes when the user chooses storage_mode="ram".
    """

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
    The lazy case: wraps a live h5py.Dataset instead of a numpy array.
    Reading `.get()` triggers an actual disk read at that moment -- the
    data is NOT loaded until something asks for it.

    Important: this only works as long as the underlying h5py.File stays
    open. ProjectObject is responsible for keeping a reference to that
    open file alive for as long as any Hdf5DiskHandle from it exists
    (see ProjectObject._open_resource in project.py) -- otherwise you'd
    get an error trying to read from a closed file.
    """

    def __init__(self, dataset):
        self._dataset = dataset

    def get(self, index=None) -> np.ndarray:
        # h5py's own indexing rule: dataset[()] reads everything,
        # dataset[some_slice] reads only that slice from disk.
        return self._dataset[()] if index is None else self._dataset[index]

    @property
    def shape(self) -> tuple:
        return self._dataset.shape

    @property
    def storage_mode(self) -> str:
        return "disk"

    def materialize(self) -> RamHandle:
        """
        Read the ENTIRE dataset into memory right now and return it as a
        RamHandle. This is how a user switches a project from disk mode
        to RAM mode mid-pipeline (see DataInterface.materialize_to_ram
        in data_interface.py) -- it's a one-way, explicit, on-demand
        conversion, never automatic.
        """
        return RamHandle(self._dataset[()])
