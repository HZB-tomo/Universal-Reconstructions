"""
Flat/dark-field correction: turns a raw detector reading into a
transmission ratio, (raw - dark) / (flat - dark). This is the first
step of preprocessing; zinger and ring removal run on its output.

Does not apply -log() here; that happens at or inside reconstruction.

No optional dependency: the math is plain numpy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np

from core.data_interface import DataInterface
from core.project import ProjectObject


class MissingFlatFieldError(Exception):
    """Raised when a project has no flat field data."""


@dataclass
class FlatDarkReference:
    """
    Reduced flat/dark reference for one project, computed once by
    build_reference() and reused across normalize() calls.
    """

    flat: np.ndarray  # reduced 2D image, shape (height, width)
    dark: Union[np.ndarray, float]  # reduced 2D image, or a scalar offset if no dark field exists


def build_reference(
    project: ProjectObject,
    reducer: Literal["mean", "median"] = "mean",
    dark_offset: Optional[float] = None,
    already_normalized: bool = False,
) -> FlatDarkReference:
    """
    Reduce a project's flat/dark stacks to the reference normalize()
    needs. Call once per project.

    `already_normalized=True` is for data whose projections were already
    flat/dark corrected upstream (e.g. ILL neutron projections): flats
    and darks are ignored (and need not exist), and the reference is a
    constant "air level" (dark 0, flat = high percentile of the
    projections), so normalize() just scales the data to about 1.0
    instead of dividing by a flat that no longer matches it.

    Raises MissingFlatFieldError if the project has no flat field data.

    Dark field resolution, in priority order:
      1. `dark_offset`, if given, is used regardless of whether the
         project has real dark frames: build_reference(project, dark_offset=100.0).
      2. Otherwise, real dark frames are reduced the same way flats are.
      3. Otherwise, `dark` is 0.0.
    """
    di = DataInterface(project)

    if already_normalized:
        n_angles, height, width = di.shape_of("projections")
        step = max(1, n_angles // 20)
        sample = di.get_projections(index=(slice(0, None, step), slice(None), slice(None)))
        air_level = float(np.percentile(sample, 98))
        return FlatDarkReference(flat=np.full((height, width), air_level, dtype=np.float32), dark=0.0)

    flats = di.get_flats()
    if flats is None:
        raise MissingFlatFieldError(
            "Cannot normalize -- this project has no flat field data, and there's no fallback for "
            "that (unlike dark field, which can default to a scalar offset)."
        )

    reduce_fn = np.mean if reducer == "mean" else np.median
    flat = reduce_fn(flats, axis=0).astype(np.float32)

    dark: Union[np.ndarray, float]
    if dark_offset is not None:
        dark = float(dark_offset)
    else:
        darks = di.get_darks()
        if darks is None:
            dark = 0.0
        else:
            dark = reduce_fn(darks, axis=0).astype(np.float32)

    return FlatDarkReference(flat=flat, dark=dark)


def normalize(raw: np.ndarray, reference: FlatDarkReference) -> np.ndarray:
    """
    Apply (raw - dark) / (flat - dark). `raw` may be a single sinogram
    (n_angles, width), a batch of rows (n_angles, batch, width), or a
    full volume (n_angles, height, width), as long as
    `reference.flat`/`.dark` are already sliced to match; see
    normalize_slice()/normalize_rows() below.

    Casts to float32 first, since raw detector data is often an
    unsigned integer type and (raw - dark) can go negative.
    """
    raw = raw.astype(np.float32, copy=False)
    return (raw - reference.dark) / (reference.flat - reference.dark)


def _dark_rows(dark: Union[np.ndarray, float], rows):
    """Slice `dark` to match `rows`, unless it is a scalar offset."""
    return dark[rows] if isinstance(dark, np.ndarray) else dark


def normalize_slice(project: ProjectObject, row: int, reference: FlatDarkReference) -> np.ndarray:
    """Normalize a single sinogram: detector row `row`, every angle."""
    di = DataInterface(project)
    raw = di.get_projections(index=(slice(None), row, slice(None)))
    row_reference = FlatDarkReference(flat=reference.flat[row], dark=_dark_rows(reference.dark, row))
    return normalize(raw, row_reference)


def normalize_rows(project: ProjectObject, rows, reference: FlatDarkReference) -> np.ndarray:
    """
    Normalize a batch of rows at once. `rows` is anything valid for
    numpy indexing along the row axis (a slice, a list of indices,
    etc.). Use `rows=slice(None)` for the entire volume; use
    normalize_slice() for a single row, since indexing with an int
    versus a length-1 slice changes the output shape.
    """
    di = DataInterface(project)
    raw = di.get_projections(index=(slice(None), rows, slice(None)))
    batch_reference = FlatDarkReference(flat=reference.flat[rows], dark=_dark_rows(reference.dark, rows))
    return normalize(raw, batch_reference)
