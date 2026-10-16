"""
Spatial binning of projection, flat and dark frames. Pure numpy.

bin_frames() bins a (H, W) image or an (N, H, W) stack. Sizes that are
not divisible by the bin factor are cropped symmetrically first, which
keeps the center of rotation (measured from the image center) unchanged.
"""
from __future__ import annotations

import numpy as np


def crop_offsets(size: int, factor: int) -> tuple[int, int]:
    """(start, stop) of the centered crop that makes `size` divisible by `factor`."""
    rem = size % factor
    start = rem // 2
    return start, size - (rem - start)


def binned_shape(h: int, w: int, bin_y: int, bin_x: int) -> tuple[int, int]:
    """Output (height, width) of bin_frames() for an (h, w) image."""
    y0, y1 = crop_offsets(h, bin_y)
    x0, x1 = crop_offsets(w, bin_x)
    return (y1 - y0) // bin_y, (x1 - x0) // bin_x


def bin_frames(frames: np.ndarray, bin_y: int = 1, bin_x: int = 1, method: str = "mean") -> np.ndarray:
    """
    Bin a (H, W) image or an (N, H, W) stack by (bin_y, bin_x).
    `method` is "mean" or "sum". Returns float32.
    """
    if method not in ("mean", "sum"):
        raise ValueError(f"unknown binning method: {method!r}")
    single = frames.ndim == 2
    if single:
        frames = frames[None]
    n, h, w = frames.shape
    y0, y1 = crop_offsets(h, bin_y)
    x0, x1 = crop_offsets(w, bin_x)
    hb, wb = (y1 - y0) // bin_y, (x1 - x0) // bin_x
    if hb == 0 or wb == 0:
        raise ValueError(f"bin factor ({bin_y}, {bin_x}) is larger than image size ({h}, {w})")
    reshaped = frames[:, y0:y1, x0:x1].reshape(n, hb, bin_y, wb, bin_x)
    acc = reshaped.sum(axis=(2, 4), dtype=np.float64)
    if method == "mean":
        acc /= bin_y * bin_x
    out = acc.astype(np.float32)
    return out[0] if single else out


def resolve_dtype(src_dtype, method: str, requested: str = "auto") -> np.dtype:
    """
    Output dtype for binned data. "auto" keeps the source dtype, except
    that a "sum" of integers becomes float32 and half floats become float32.
    """
    if requested != "auto":
        return np.dtype(requested)
    src = np.dtype(src_dtype)
    if np.issubdtype(src, np.floating):
        return src if src.itemsize >= 4 else np.dtype(np.float32)
    return src if method == "mean" else np.dtype(np.float32)


def cast_to_dtype(arr: np.ndarray, dtype) -> np.ndarray:
    """Cast to `dtype`, rounding and clipping when the target is an integer type."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        arr = np.clip(np.rint(arr), info.min, info.max)
    return arr.astype(dtype)
