"""
Browsing a reconstructed volume stored in an HDF5 file one section at a
time, for ImageViewer.show_stack_lazy(). Qt-free.
"""
from __future__ import annotations

from typing import Callable

import h5py
import numpy as np


def open_volume(path: str, dataset_path: str = "/volume") -> tuple[Callable[[object], np.ndarray], tuple]:
    """(getter, shape). `getter(index)` takes any numpy-style index
    (an int slice number, or e.g. `(slice(None), j)` for an xz section)
    and reads just that from disk. The file is opened per read rather
    than held open, so it can be moved, deleted, or overwritten by a
    later export while the viewer still shows it."""
    with h5py.File(path, "r") as f:
        shape = tuple(f[dataset_path].shape)

    def get(index) -> np.ndarray:
        with h5py.File(path, "r") as f:
            return f[dataset_path][index]

    return get, shape
