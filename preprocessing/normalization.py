"""
Flat/dark-field correction: turning a raw detector reading into a
transmission ratio, (raw - dark) / (flat - dark). This is the first
step of preprocessing -- everything else (zinger removal, ring
removal, ...) runs on its output.

Deliberately does NOT apply -log() here. Every real pipeline we checked
(see oldscripts/ -- BAMline_recon_v0p2.py, full_reconstruction_cli.py,
Reconstruction_algotom_loop.ipynb, Recon_algotom.py) applies -log()
right before/inside reconstruction, never as part of flat/dark
correction -- ring removal and phase retrieval both run on the
non-logged transmission data these functions produce.

No optional dependency here -- the math is simple enough that it would
be silly to require an external package just for this (unlike
corrections.py, which genuinely needs one).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np

from core.data_interface import DataInterface
from core.project import ProjectObject


class MissingFlatFieldError(Exception):
    """
    Raised when there's no flat field data to normalize against. Unlike
    a missing dark field (which has a reasonable fallback -- see
    build_reference()), there's no substitute for a missing flat field,
    so this is a hard stop rather than a default.
    """


@dataclass
class FlatDarkReference:
    """
    The reduced flat/dark reference for one project, computed once by
    build_reference() and reused across every normalize() call --
    recomputing the mean of the flats stack on every single-slice call
    would redo the same work over and over for no reason.
    """

    flat: np.ndarray  # reduced 2D image, shape (height, width)
    dark: Union[np.ndarray, float]  # reduced 2D image, or a scalar offset if no dark field exists


def build_reference(
    project: ProjectObject,
    reducer: Literal["mean", "median"] = "mean",
    dark_offset: Optional[float] = None,
) -> FlatDarkReference:
    """
    Reduce this project's flat/dark stacks down to the single reference
    normalize() actually needs. Call this once per project.

    Flats are mandatory -- raises MissingFlatFieldError if the project
    has none, since there's no reasonable substitute for a missing flat
    field. Darks are optional, with three cases, in priority order:

      1. `dark_offset` given -- always wins, regardless of whether the
         project also has real dark frames. This is for exactly the
         oldscripts/BAMline_recon_v0p2.py situation: no dark field was
         ever acquired, but a known constant detector offset (there,
         hardcoded as 100) stands in for it. Pass it explicitly:
         build_reference(project, dark_offset=100.0).
      2. No `dark_offset`, but the project has real dark frames --
         reduced the same way flats are.
      3. Neither -- `dark` becomes a scalar 0.0 (no dark correction at
         all), the same as case 1 with dark_offset=0.0.
    """
    di = DataInterface(project)

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
    The actual formula: (raw - dark) / (flat - dark). Pure numpy
    broadcasting, nothing shape-specific about it -- `raw` can be a
    single sinogram (n_angles, width), a batch of rows
    (n_angles, batch, width), or the whole volume
    (n_angles, height, width), as long as `reference.flat`/`.dark` are
    already sliced to match (see normalize_slice()/normalize_rows()
    below, which handle that bookkeeping for you via DataInterface).

    Casts to float32 first: raw detector data is often an unsigned
    integer type, and (raw - dark) needs to be able to go negative.
    """
    raw = raw.astype(np.float32, copy=False)
    return (raw - reference.dark) / (reference.flat - reference.dark)


def _dark_rows(dark: Union[np.ndarray, float], rows):
    """Slice `dark` to match `rows`, unless it's just a scalar offset (no per-pixel indexing to do)."""
    return dark[rows] if isinstance(dark, np.ndarray) else dark


def normalize_slice(project: ProjectObject, row: int, reference: FlatDarkReference) -> np.ndarray:
    """Normalize a single sinogram -- detector row `row`, every angle."""
    di = DataInterface(project)
    raw = di.get_projections(index=(slice(None), row, slice(None)))
    row_reference = FlatDarkReference(flat=reference.flat[row], dark=_dark_rows(reference.dark, row))
    return normalize(raw, row_reference)


def normalize_rows(project: ProjectObject, rows, reference: FlatDarkReference) -> np.ndarray:
    """
    Normalize a batch of rows at once -- `rows` is anything valid for
    numpy indexing along the row axis (a slice, a list of indices, ...).
    Pass `rows=slice(None)` for the entire volume in one call; pass a
    single int to normalize_slice() instead for one sinogram, since
    slicing with an int vs. a length-1 slice changes the output shape.
    """
    di = DataInterface(project)
    raw = di.get_projections(index=(slice(None), rows, slice(None)))
    batch_reference = FlatDarkReference(flat=reference.flat[rows], dark=_dark_rows(reference.dark, rows))
    return normalize(raw, batch_reference)
