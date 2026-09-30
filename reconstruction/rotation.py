"""
Rotating a reconstructed volume about the scan axis (every xy slice,
uniformly) by relabeling the sinogram's angle axis instead of
resampling the reconstructed image -- exact, and cheaper than an
image-domain rotation, when the scan's angular coverage supports it;
falls back to rotating the reconstructed volume otherwise (an
arbitrary geometry_type, or a limited-angle scan that doesn't cover a
full parallel-beam half-period or cone-beam full period).
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.ndimage import rotate as _ndi_rotate

from reconstruction.api import reconstruct
from reconstruction.backend import Geometry, ReconstructionBackend

# One angular step of slack on top of the exact 180/360 threshold, so a
# scan built as N angles evenly spaced over [0, span) (the usual
# construction, e.g. np.linspace(0, 180, n, endpoint=False)) -- whose
# last sample therefore falls one step short of `span` -- still counts
# as covering it.
_COVERAGE_SLACK = 1e-6


def _angular_step(sorted_angles_deg: np.ndarray) -> Optional[float]:
    """The common spacing between already-sorted, distinct angles, or
    None if they aren't evenly spaced."""
    diffs = np.diff(sorted_angles_deg)
    if diffs.size == 0:
        return None
    step = float(np.median(diffs))
    if step <= 0 or not np.allclose(diffs, step, rtol=1e-3, atol=1e-6):
        return None
    return step


def supports_exact_rotation(angles_deg, geometry_type: str) -> bool:
    """
    Whether shift_sinogram() can rotate this scan's sinogram exactly
    (no image-domain interpolation): the angles must be evenly spaced
    and their span must reach a full parallel-beam half-period
    (180deg, via the p(t+180,s) = p(t,-s) identity) or a full period
    (360deg -- required for "cone", since that identity is
    parallel-beam-only).
    """
    angles_deg = np.asarray(angles_deg, dtype=float)
    if angles_deg.shape[0] < 2:
        return False
    sorted_angles = np.sort(angles_deg)
    step = _angular_step(sorted_angles)
    if step is None:
        return False
    span = float(sorted_angles[-1] - sorted_angles[0])
    tol = step + _COVERAGE_SLACK
    if geometry_type == "parallel":
        return span >= 180.0 - tol
    if geometry_type == "cone":
        return span >= 360.0 - tol
    return False


def shift_sinogram(
    stack: np.ndarray, angles_deg: np.ndarray, geometry_type: str, shift_deg: float,
) -> Optional[np.ndarray]:
    """
    A new sinogram stack such that reconstructing it with the SAME
    `angles_deg` produces the input's reconstructed volume rotated by
    `shift_deg` about the scan axis -- or None if
    supports_exact_rotation() is False for this angle array/geometry
    (caller should fall back to rotating the reconstructed volume
    instead; see reconstruct_rotated()).

    `stack` is (n_angles, n_slices, width), angle-first -- the shape
    reconstruct() and preprocessing.normalize_rows() use. `shift_deg`
    need not land on the angular sampling grid: off-grid values are
    linearly interpolated along the angle axis.
    """
    angles_deg = np.asarray(angles_deg, dtype=float)
    if not supports_exact_rotation(angles_deg, geometry_type):
        return None

    order = np.argsort(angles_deg)
    sorted_angles = angles_deg[order]
    sorted_stack = stack[order]
    n = sorted_angles.shape[0]
    step = _angular_step(sorted_angles)
    span = float(sorted_angles[-1] - sorted_angles[0])
    tol = step + _COVERAGE_SLACK

    if span >= 360.0 - tol:
        # A full period was actually acquired -- true of any
        # geometry_type once coverage reaches 360deg.
        period_stack, n_period = sorted_stack, n
    else:
        # geometry_type == "parallel" and 180deg <= span < 360deg:
        # extend to a full 360deg period via the parallel-beam-only
        # flip identity p(t+180, s) = p(t, -s).
        period_stack = np.concatenate([sorted_stack, sorted_stack[:, :, ::-1]], axis=0)
        n_period = 2 * n

    shift_idx = shift_deg / step
    a = np.arange(n)
    idx = (a - shift_idx) % n_period
    idx0 = np.floor(idx + 1e-9).astype(int) % n_period
    idx1 = (idx0 + 1) % n_period
    frac = (idx - np.floor(idx + 1e-9)).astype(stack.dtype)[:, None, None]
    shifted_sorted = (1.0 - frac) * period_stack[idx0] + frac * period_stack[idx1]

    result = np.empty_like(stack)
    result[order] = shifted_sorted.astype(stack.dtype)
    return result


def reconstruct_rotated(
    sinogram: np.ndarray,
    angles: np.ndarray,
    center,
    geometry: Geometry,
    backend: Union[str, ReconstructionBackend],
    rotate_deg: float,
    device: Optional[str] = None,
    **options,
) -> np.ndarray:
    """
    reconstruct(), rotated by `rotate_deg` degrees about the scan axis
    (every xy slice, uniformly) -- same call shape as reconstruct()
    (one sinogram or a stack; see its docstring).

    Uses shift_sinogram() -- reconstructing shifted sinogram data
    instead of resampling the reconstructed image -- whenever
    supports_exact_rotation() allows it; otherwise reconstructs
    normally and rotates the resulting volume with
    scipy.ndimage.rotate (bilinear, no reshape -- pixels rotated
    outside the original frame are lost, matching the frame
    shift_sinogram()'s exact path also can't recover).
    """
    sinogram = np.asarray(sinogram)
    was_single = sinogram.ndim == 2
    stack = sinogram[:, np.newaxis, :] if was_single else sinogram
    angles = np.asarray(angles, dtype=float)

    shifted_stack = shift_sinogram(stack, angles, geometry.geometry_type, rotate_deg)
    if shifted_stack is not None:
        shifted = shifted_stack[:, 0, :] if was_single else shifted_stack
        return reconstruct(shifted, angles, center, geometry, backend, device, **options)

    volume = reconstruct(sinogram, angles, center, geometry, backend, device, **options)
    axes = (0, 1) if volume.ndim == 2 else (1, 2)
    return _ndi_rotate(volume, angle=rotate_deg, axes=axes, reshape=False, order=1, mode="constant")
