"""
Turns a completion-ready ProjectObject's raw projections into
reconstruction-ready sinograms: flat/dark correction first, then
artifact removal. Does not apply -log(); that happens at or inside
reconstruction.

  normalization.py   flat/dark correction. Pure numpy, no optional
                      dependency.
  corrections.py     zinger/ring-artifact removal. Requires the
                      optional `algotom` package; raises ImportError
                      with an install hint if it is missing.

Both operate on a single sinogram (one detector row, every angle), a
batch of rows, or the whole volume.
"""
from preprocessing.corrections import process_volume, remove_ring_artifacts, remove_zinger
from preprocessing.normalization import (
    FlatDarkReference,
    MissingFlatFieldError,
    build_reference,
    normalize,
    normalize_rows,
    normalize_slice,
)

__all__ = [
    "FlatDarkReference",
    "MissingFlatFieldError",
    "build_reference",
    "normalize",
    "normalize_slice",
    "normalize_rows",
    "remove_zinger",
    "remove_ring_artifacts",
    "process_volume",
]
