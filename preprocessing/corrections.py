"""
Zinger and ring-artifact removal, applied to flat/dark-corrected
sinograms (normalization.py's output).

Requires the optional `algotom` package
(https://github.com/algotomography/algotom). Each function imports it
lazily and raises ImportError with an install hint if it is missing.
Install with `pip install algotom` or `pip install .[preprocessing]`.

process_volume() applies a single-sinogram function to every row of a
projection volume in parallel, via algotom's parallel_process_slices(),
which operates on axis=1 (the row axis).
"""
from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np

_INSTALL_HINT = "pip install algotom"


def _algotom_removal():
    try:
        import algotom.prep.removal as remo
    except ImportError as e:
        raise ImportError(
            f"Zinger/ring-artifact removal requires the 'algotom' package, which isn't installed. "
            f"Install it with: {_INSTALL_HINT}"
        ) from e
    return remo


def _algotom_utility():
    try:
        import algotom.util.utility as util
    except ImportError as e:
        raise ImportError(
            f"Batch processing (process_volume) requires the 'algotom' package, which isn't installed. "
            f"Install it with: {_INSTALL_HINT}"
        ) from e
    return util


def remove_zinger(sinogram: np.ndarray, threshold: float = 0.08, size: int = 2) -> np.ndarray:
    """
    Remove isolated outlier pixels (zingers) from one sinogram. Wraps
    algotom.prep.removal.remove_zinger().
    """
    remo = _algotom_removal()
    return remo.remove_zinger(sinogram, threshold, size=size)


RingMethod = Literal[
    "sorting", "filtering", "fitting", "normalization", "all",
    "large_stripe", "dead_stripe", "interpolation", "2d_filtering_sorting",
    "fft", "wavelet_fft", "regularization",
]

_RING_METHODS = {
    "sorting": "remove_stripe_based_sorting",
    "filtering": "remove_stripe_based_filtering",
    "fitting": "remove_stripe_based_fitting",
    "normalization": "remove_stripe_based_normalization",
    "all": "remove_all_stripe",
    "large_stripe": "remove_large_stripe",
    "dead_stripe": "remove_dead_stripe",
    "interpolation": "remove_stripe_based_interpolation",
    "2d_filtering_sorting": "remove_stripe_based_2d_filtering_sorting",
    "fft": "remove_stripe_based_fft",
    "wavelet_fft": "remove_stripe_based_wavelet_fft",
    "regularization": "remove_stripe_based_regularization",
}


def remove_ring_artifacts(sinogram: np.ndarray, method: RingMethod = "sorting", **kwargs) -> np.ndarray:
    """
    Remove ring (stripe) artifacts from one sinogram, using one of
    algotom's stripe-removal algorithms. `"all"` runs algotom's
    combined method. Extra keyword arguments are passed to the chosen
    algotom function; see algotom.prep.removal for what each accepts.
    """
    remo = _algotom_removal()
    fn = getattr(remo, _RING_METHODS[method])
    return fn(sinogram, **kwargs)


def process_volume(raw: np.ndarray, fn: Callable[..., np.ndarray], *args: Any, ncore: int = None, **kwargs: Any) -> np.ndarray:
    """
    Apply a single-sinogram function (remove_zinger,
    remove_ring_artifacts, or any other per-sinogram callable) to every
    row of a projection volume in parallel. `*args`/`**kwargs` are
    forwarded to `fn` for every row:

        process_volume(raw, remove_zinger, 0.08, size=2)
        process_volume(raw, remove_ring_artifacts, method="sorting")

    `fn` is wrapped in a closure that places the sinogram first, since
    algotom's parallel_process_slices() supplies the sinogram as its
    own leading argument, and reserves the parameter name `method`
    for itself. Closures are not picklable, so this only works with
    algotom's default `prefer="threads"`, not `prefer="processes"`.
    """
    util = _algotom_utility()

    def bound(sinogram):
        return fn(sinogram, *args, **kwargs)

    return util.parallel_process_slices(raw, bound, [], axis=1, ncore=ncore)
