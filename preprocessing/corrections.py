"""
Zinger and ring-artifact removal, on flat/dark-corrected sinograms
(normalization.py's output) -- both run before any -log()/reconstruction.

Both need the `algotom` package (https://github.com/algotomography/algotom),
which is deliberately NOT a mandatory dependency of this project -- we
don't want every user of this software to install a package they only
need if they're doing artifact removal. Every function below imports
algotom lazily, inside the function, and raises a plain ImportError
with an actionable message ("pip install algotom") if it's missing,
instead of this whole module failing to import for everyone. If you
want these functions, `pip install algotom` (or `pip install .[preprocessing]`
from the project root) once and they work exactly as documented.

`algotom` (not `sarepy`) is the one used here: oldscripts/full_reconstruction_cli.py
tracks specific `algotom` version behavior in its comments and installs
from PyPI, while oldscripts/BAMline_recon_v0p2.py's `sarepy` import comes
from a hardcoded local Windows path, not a real package install.

Unlike normalize() (core/normalization.py), these can't be vectorized
into one broadcastable array op -- sorting/filtering/fitting a
sinogram's rows or columns is inherently a per-2D-array algorithm, not
elementwise math. So "batch" here means "run the same single-sinogram
function once per row, in parallel" -- process_volume() below delegates
that directly to algotom's own parallel_process_slices(), which already
assumes axis=1 is the row axis, matching this codebase's own
(n_angles, height, width) convention.
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
    Remove isolated outlier pixels (zingers -- cosmic ray hits, detector
    glitches) from one sinogram. Thin wrapper over
    algotom.prep.removal.remove_zinger(); `threshold=0.08` matches the
    value used in oldscripts/full_reconstruction_cli.py, not a
    universally-validated default -- tune it for your data.
    """
    remo = _algotom_removal()
    return remo.remove_zinger(sinogram, threshold, size=size)


RingMethod = Literal["sorting", "filtering", "fitting", "normalization", "all"]

# The four interchangeable stripe-removal algorithms algotom provides
# (the standard Vo et al. taxonomy), plus "all" for algotom's own
# combined method. "normalization" here is one algorithm's name, not
# related to normalization.py's flat/dark correction -- it renormalizes
# sinogram columns against a smoothed reference derived from the
# sinogram's own statistics, entirely self-contained inside algotom.
_RING_METHODS = {
    "sorting": "remove_stripe_based_sorting",
    "filtering": "remove_stripe_based_filtering",
    "fitting": "remove_stripe_based_fitting",
    "normalization": "remove_stripe_based_normalization",
    "all": "remove_all_stripe",
}


def remove_ring_artifacts(sinogram: np.ndarray, method: RingMethod = "sorting", **kwargs) -> np.ndarray:
    """
    Remove ring (stripe) artifacts from one sinogram, using one of
    algotom's interchangeable algorithms. `method="sorting"` is the
    fastest/simplest; `"all"` runs algotom's own combined method,
    generally the most thorough. Extra keyword arguments (sigma, size,
    order, ...) are passed straight through to the chosen algotom
    function -- see algotom.prep.removal for what each one accepts.
    """
    remo = _algotom_removal()
    fn = getattr(remo, _RING_METHODS[method])
    return fn(sinogram, **kwargs)


def process_volume(raw: np.ndarray, fn: Callable[..., np.ndarray], *args: Any, ncore: int = None, **kwargs: Any) -> np.ndarray:
    """
    Apply a single-sinogram function -- remove_zinger, remove_ring_artifacts,
    or any other per-sinogram callable, including algotom's own
    functions directly -- to every row of a projection volume/batch at
    once, in parallel. `*args`/`**kwargs` are forwarded to `fn` for
    every row, e.g.:

        process_volume(raw, remove_zinger, 0.08, size=2)
        process_volume(raw, remove_ring_artifacts, method="sorting")

    `fn` is wrapped in a closure that puts the sinogram first, rather
    than forwarding `*args`/`**kwargs` to algotom's call directly --
    two independent reasons this can't just be a functools.partial:
    parallel_process_slices() supplies the sinogram as its OWN leading
    positional argument, whereas functools.partial's pre-bound args
    come first and the call-time argument gets appended after them
    (the wrong order for fn(sinogram, *args, **kwargs)); separately,
    parallel_process_slices() has its own reserved parameter names
    (data, method, parameters, axis, ncore, prefer), and "method" in
    particular collides with remove_ring_artifacts()'s own `method`
    argument if forwarded into that same call directly. The closure
    below sidesteps both issues, regardless of what `fn`'s own
    parameter names happen to be. One tradeoff: closures aren't
    picklable, so this only works with algotom's default
    prefer="threads", not prefer="processes".
    """
    util = _algotom_utility()

    def bound(sinogram):
        return fn(sinogram, *args, **kwargs)

    return util.parallel_process_slices(raw, bound, [], axis=1, ncore=ncore)
