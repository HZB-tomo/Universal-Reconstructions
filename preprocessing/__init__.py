"""
preprocessing/ turns a completion-ready ProjectObject's raw projections
into reconstruction-ready sinograms: flat/dark correction first, then
whatever artifact removal the data needs, all still upstream of any
-log() (that belongs to reconstruction, not here -- see
normalization.py's module docstring for why).

Two files, two different kinds of dependency:

  normalization.py   flat/dark correction -- pure numpy, no optional
                      dependency, always available.
  corrections.py     zinger/ring-artifact removal -- needs the
                      `algotom` package, which is deliberately NOT a
                      mandatory dependency of this project (see
                      corrections.py's module docstring). Calling a
                      function here without algotom installed raises a
                      clear ImportError telling you what to install,
                      rather than this whole package failing to import.

Both operate at the same granularity: one sinogram (a fixed detector
row, every angle), a batch of rows, or the whole volume -- see each
module for how that's handled.
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
