"""
find_center() / find_tilt_and_center(): the public entry points for
center-of-rotation and rotation-axis-tilt calibration.

Both take plain arrays, not a ProjectObject -- same reasoning as
reconstruction.reconstruct() (see that module's docstring): the caller
prepares whatever normalized data this needs beforehand, the same way
it already calls preprocessing.normalize_rows() itself.

find_tilt_and_center() is the one that matters most in practice: it
finds the center of rotation independently at several rows (via
whichever backend's find_center() -- always the same deterministic,
sinogram-domain operation, no reconstruction involved anywhere in this
module) and fits a line through the results. This is the same method
used by real reference code
(oldscripts/Reconstruction_algotom_loop.ipynb's own
find_tilt_and_center()), and it was verified directly against this
project's own phantom generator before being adopted here, not assumed
to work by analogy:

  - A hand-constructed, KNOWN rotation-axis tilt (axis lean: the
    physical rotation axis not parallel to the detector's row axis, so
    the effective center of rotation drifts linearly with row) was
    recovered to within about 1% of its true slope/angle this way.
  - A DIFFERENT effect -- an in-plane rotation of each projection frame
    ("detector roll") -- was verified NOT to be recoverable by this
    method at all: a simulated 1-degree roll produced a recovered
    "tilt" of roughly -43 degrees, nonsense. Roll mixes row and column
    information (a fixed physical height smears across several detector
    rows depending on column position), which breaks the per-row
    independence this method relies on.

So this module only ever targets axis lean, matching what the real
reference code above also targets -- detector-roll detection/correction
is a genuinely different, harder problem and is explicitly out of scope
here. tests/fixtures/realistic_phantom's `tilt_deg` parameter simulates
axis lean specifically for this reason (see simulate.py's
apply_cor_and_tilt() docstring) -- it used to simulate detector roll,
which this module could never have been validated against correctly.

This orchestration -- picking rows, the robust line fit -- is
deliberately NOT part of the backend contract (calibration/backend.py):
every backend only has to implement the one, single-sinogram
find_center() primitive; a from-scratch reimplementation (if algotom is
ever unavailable) gets tilt-finding for free, unmodified.

A manual/visual center check (reconstruct a range of candidate centers
and look at them) deliberately has no function here -- it's just a
direct call to reconstruction.reconstruct() with an array of candidate
centers, exactly as oldscripts/Reconstruction_algotom_loop.ipynb's own
center_visual_scan() does. No metric-based automatic scoring for that
path: verified directly (see this project's own investigation, not
covered further here) that naive image-quality metrics on a
CPU-fallback FBP reconstruction are unreliable at sub-pixel granularity
-- the deterministic sinogram-domain methods in this module are the
trustworthy automatic path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from calibration.backend import CalibrationBackend, get_backend


def find_center(
    sino_180: np.ndarray,
    backend: Union[str, CalibrationBackend],
    start: Optional[float] = None,
    stop: Optional[float] = None,
    **options,
) -> float:
    """
    Find the center of rotation of a single 180-degree sinogram,
    (n_angles, width).

    `backend` selects which plugin does the work -- a name to look up
    via get_backend(), or a CalibrationBackend instance directly.
    Always explicit, same as reconstruction.reconstruct()'s `backend`.

    `start`/`stop` bound the search range in column-index units; if
    either is omitted, defaults to `width/2 +/- width/4` -- wider than
    a backend's own default (algotom's is `width/2 +/- width/16`,
    confirmed too narrow for anything beyond a small COR deviation),
    but NOT the full column range: verified directly that widening all
    the way to (0, width) makes the underlying search markedly less
    reliable (it starts finding the wrong minimum on perfectly good
    rows, not just failing safely) rather than more robust. A quarter
    of the width is a reasonable, but not sacred, middle ground -- pass
    `start`/`stop` explicitly if your setup's real deviations are
    larger than that.
    """
    backend_instance = get_backend(backend) if isinstance(backend, str) else backend
    sino_180 = np.asarray(sino_180)
    width = sino_180.shape[-1]
    if start is None:
        start = width / 2.0 - width / 4.0
    if stop is None:
        stop = width / 2.0 + width / 4.0
    return backend_instance.find_center(sino_180, start=start, stop=stop, **options)


@dataclass
class TiltAndCenterResult:
    """
    Everything find_tilt_and_center() found, not just the two headline
    numbers -- the per-row estimates and which ones were trusted are
    worth keeping to diagnose a bad fit, the same way
    Reconstruction_algotom_loop.ipynb's own version prints them.
    """

    tilt_deg: float  # arctan(pixels of center drift per row) -- see module docstring
    center: float  # center of rotation at the middle row
    rows: np.ndarray = field(repr=False)  # which rows were sampled
    centers: np.ndarray = field(repr=False)  # this backend's per-row estimate, one per row in `rows`
    used: np.ndarray = field(repr=False)  # boolean mask into rows/centers -- False for rows dropped as outliers


def find_tilt_and_center(
    projections: np.ndarray,
    backend: Union[str, CalibrationBackend],
    num_rows: int = 12,
    margin: int = 0,
    outlier_tolerance: float = 20.0,
    **options,
) -> TiltAndCenterResult:
    """
    Find rotation-axis tilt and center-of-rotation together from a full
    projection stack, (n_angles, n_rows, width) -- angle-first, matching
    preprocessing.normalize_rows()'s own output, so its result can be
    handed to this function directly, no reshaping.

    Method (see module docstring for why this is the right one, and
    what it deliberately does NOT detect): sample `num_rows` rows
    spread across the stack (skipping `margin` rows at the top/bottom,
    where real scan data is often absent or unreliable), call
    find_center() independently on each row's own 180-degree sinogram,
    drop rows whose estimate is more than `outlier_tolerance` pixels
    from the median (a row with no actual sample material gives a
    meaningless estimate), then fit a line (center vs. row index)
    through what's left.

    `backend`/`**options` are forwarded to find_center() for every row.

    Raises RuntimeError if fewer than 3 rows survive the outlier check
    -- not enough left to trust a line fit.
    """
    projections = np.asarray(projections)
    n_angles, n_rows, width = projections.shape
    if n_rows - 2 * margin < 2:
        raise ValueError(f"Not enough rows ({n_rows}) left after applying margin={margin} on both sides.")

    rows = np.linspace(margin, n_rows - 1 - margin, num_rows).astype(int)
    centers = np.array(
        [find_center(projections[:, r, :], backend=backend, **options) for r in rows],
        dtype=float,
    )

    median = np.median(centers)
    used = np.abs(centers - median) < outlier_tolerance
    if used.sum() < 3:
        raise RuntimeError(
            f"Center estimates are too inconsistent to fit a line: {np.round(centers, 2).tolist()}. "
            "Check the search range, or whether this is really a standard scan "
            "(not an offset-axis/extended-field-of-view one -- see find_center_360 for that case)."
        )

    slope, intercept = np.polyfit(rows[used], centers[used], 1)
    tilt_deg = float(np.degrees(np.arctan(slope)))
    center = float(slope * ((n_rows - 1) / 2.0) + intercept)

    return TiltAndCenterResult(tilt_deg=tilt_deg, center=center, rows=rows, centers=centers, used=used)
