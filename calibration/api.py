"""
Public entry points for center-of-rotation and rotation-axis-tilt
calibration: find_center(), find_tilt(), centers_for_rows().

find_center() finds the center of rotation of one sinogram, typically
the middle row of a projection stack. find_tilt() finds rotation-axis
tilt (axis lean) using several rows. centers_for_rows() turns a tilt
angle and a center into a per-row center array for reconstruction.

Rotation-axis tilt means the physical rotation axis is not parallel to
the detector's row axis, so the true center of rotation drifts linearly
with row index. It is not the same as an in-plane rotation of each
projection frame; find_tilt() only detects axis lean.

None of these functions take a ProjectObject. The caller prepares
normalized data beforehand, the same way it calls
preprocessing.normalize_rows().
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
    shape (n_angles, width).

    `backend` is a registered backend name or a CalibrationBackend
    instance.

    `start`/`stop` bound the search range in column-index units.
    Defaults to `width/2 +/- width/4` if omitted. Pass `start`/`stop`
    explicitly for offsets larger than a quarter of the width.
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
class TiltResult:
    """
    Result of find_tilt().

    tilt_deg: rotation-axis tilt, in degrees.
    reference_row: the row center_at_reference_row is reported at.
    center_at_reference_row: center of rotation at reference_row, from
        the same line fit that produced tilt_deg.
    rows: row indices sampled.
    centers: per-row center estimate for each row in `rows`.
    used: boolean mask into rows/centers; False marks a row dropped as
        an outlier.
    """

    tilt_deg: float
    reference_row: float
    center_at_reference_row: float
    rows: np.ndarray = field(repr=False)
    centers: np.ndarray = field(repr=False)
    used: np.ndarray = field(repr=False)


def find_tilt(
    projections: np.ndarray,
    backend: Union[str, CalibrationBackend],
    center: Optional[float] = None,
    num_rows: int = 12,
    margin: int = 0,
    outlier_tolerance: float = 20.0,
    **options,
) -> TiltResult:
    """
    Find rotation-axis tilt from a projection stack, shape
    (n_angles, n_rows, width).

    Method: sample `num_rows` rows spread evenly across the stack,
    skipping `margin` rows at the top and bottom. Call find_center()
    independently on each sampled row's sinogram. Drop any row whose
    estimate is more than `outlier_tolerance` pixels from the median of
    all estimates. Fit a line (center vs. row index) through the
    remaining rows. The slope gives tilt_deg; the line's value at the
    stack's middle row is returned as center_at_reference_row.

    `center`, if given, is an already-known center of rotation
    (typically from find_center() at the stack's middle row). Each
    row's search window is re-centered on `center +/- width/4` instead
    of the sinogram's nominal midpoint, before searching. This affects
    only rows where the true center falls outside the default window;
    it has no effect when the offset is already within range. Pass
    `start`/`stop` explicitly to override the window directly.

    `backend`/`**options` are forwarded to find_center() for every
    sampled row.

    Raises ValueError if fewer than 2 rows remain after applying
    `margin`. Raises RuntimeError if fewer than 3 rows survive the
    outlier check.
    """
    projections = np.asarray(projections)
    n_angles, n_rows, width = projections.shape
    if n_rows - 2 * margin < 2:
        raise ValueError(f"Not enough rows ({n_rows}) left after applying margin={margin} on both sides.")

    if center is not None and "start" not in options and "stop" not in options:
        options = dict(options, start=center - width / 4.0, stop=center + width / 4.0)

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
            "Check the search range, or whether this is an offset-axis/extended-field-of-view scan."
        )

    slope, intercept = np.polyfit(rows[used], centers[used], 1)
    tilt_deg = float(np.degrees(np.arctan(slope)))
    reference_row = (n_rows - 1) / 2.0
    center_at_reference_row = float(slope * reference_row + intercept)

    return TiltResult(
        tilt_deg=tilt_deg,
        reference_row=reference_row,
        center_at_reference_row=center_at_reference_row,
        rows=rows,
        centers=centers,
        used=used,
    )


def centers_for_rows(tilt_deg: float, center: float, reference_row: float, rows) -> np.ndarray:
    """
    Build a per-row center array from a tilt angle and a center at one
    known row, ready to pass to reconstruction.reconstruct()'s `center`
    parameter.

    `tilt_deg`, `center`, and `reference_row` may come from find_tilt(),
    find_center(), or a value supplied directly.

    `rows` is the row indices to compute centers for; it does not need
    to match the rows find_tilt() sampled, and does not need to include
    `reference_row`.
    """
    rows = np.asarray(rows, dtype=float)
    slope = np.tan(np.radians(tilt_deg))
    return center + slope * (rows - reference_row)
