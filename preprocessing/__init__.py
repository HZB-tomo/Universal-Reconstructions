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
  beam_hardening.py  polynomial linearization of beam-hardened (polychromatic)
                      attenuation, with fits from a calibration or a spectrum.
                      Pure numpy.
  phase.py           propagation-based phase retrieval (Paganin). Pure
                      numpy/scipy. A 2D filter across rows, so blocks of
                      rows need a halo; see phase_halo_rows().

Both operate on a single sinogram (one detector row, every angle), a
batch of rows, or the whole volume.
"""
from preprocessing.corrections import process_volume, remove_ring_artifacts, remove_zinger
from preprocessing.beam_hardening import (
    coefficients_from_spectrum,
    correct_beam_hardening,
    fit_polynomial,
    is_monotonic,
    valid_attenuation_limit,
)
from preprocessing.phase import (
    PhaseMethod,
    PhaseParams,
    get_phase_method,
    list_phase_methods,
    paganin_filter,
    phase_halo_rows,
    register_phase_method,
    retrieve_phase,
)
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
    "correct_beam_hardening",
    "fit_polynomial",
    "coefficients_from_spectrum",
    "is_monotonic",
    "valid_attenuation_limit",
    "retrieve_phase",
    "phase_halo_rows",
    "PhaseParams",
    "PhaseMethod",
    "register_phase_method",
    "get_phase_method",
    "list_phase_methods",
    "paganin_filter",
]
