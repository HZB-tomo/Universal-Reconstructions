"""
Turns a phantom volume (phantom.py) into realistic, detector-level raw
data: parallel-beam forward projection, center-of-rotation deviation,
rotation-axis tilt (axis lean), and a simulated flat/dark-imperfect
detector.

The forward model is the inverse of preprocessing.normalize()'s
formula, so normalize() applied to this generator's output (with the
flat/dark it also produces) recovers the ideal transmission:

    transmission = exp(-line_integral * attenuation_scale)      (Beer-Lambert)
    raw          = dark_offset + gain * flat_intensity * transmission + noise
    flat_frame   = dark_offset + gain * flat_intensity + noise
    dark_frame   = dark_offset + noise

`gain` (spatial non-uniformity, scratches, dead/hot pixels) and
`dark_offset` are the fixed detector patterns, shared between every
projection, flat, and dark frame. Ring artifacts fall out of this
fixed-pattern `gain` map being imperfectly corrected in every
projection angle; there is no separate ring-artifact simulation step.

`attenuation_scale` has no inherent physical scale of its own --
line_integral is a raw pixel-count sum, so what counts as a realistic
value depends on the object. attenuation_scale_for_min_transmission()
solves for it from the actual data instead of guessing.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi


def project_parallel(volume: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """
    Ideal, noise-free parallel-beam line integrals: rotate the volume
    around its height axis (axes 1/2, the square cross-section) by each
    angle and sum along one in-plane axis. Returns shape
    (n_angles, height, width).
    """
    projections = np.empty((len(angles_deg), *volume.shape[:2]), dtype=np.float32)
    for i, angle in enumerate(angles_deg):
        rotated = ndi.rotate(volume, -angle, axes=(1, 2), reshape=False, order=1, mode="constant", cval=0.0)
        projections[i] = rotated.sum(axis=1)
    return projections


def attenuation_scale_for_min_transmission(line_integrals: np.ndarray, target_min_transmission: float = 0.14) -> float:
    """
    Solve for the attenuation_scale (Beer-Lambert: transmission =
    exp(-line_integral * attenuation_scale)) that makes the single
    most-attenuating simulated ray's transmission equal
    target_min_transmission, the way a real CT acquisition's
    exposure/filtration is actually set up: tuned so even the densest
    path through the sample still transmits a usable fraction of the
    beam (real practice: at least ~10-15%), not so little the signal
    there is noise-dominated.

    A fixed attenuation_scale has no such physical grounding, and
    scaling it only by array width (an earlier version of this
    function did) prevents transmission underflowing to exact zero in
    float32 but does not make it realistic -- it can still land at a
    small fraction of a percent. This solves for the actual worst-case
    ray in `line_integrals` instead of guessing from shape alone, so it
    adapts to whatever object/geometry produced them.

    Returns 1.0 (arbitrary; there is nothing to calibrate against) if
    `line_integrals` is all zero, e.g. an empty phantom.
    """
    max_line_integral = float(np.max(line_integrals))
    if max_line_integral <= 0.0:
        return 1.0
    return -np.log(target_min_transmission) / max_line_integral


def apply_cor_and_tilt(projections: np.ndarray, cor_deviation: float, tilt_deg: float) -> np.ndarray:
    """
    Inject a known center-of-rotation deviation and rotation-axis tilt
    (axis lean) into ideal line integrals.

    Axis lean means the physical rotation axis is not parallel to the
    detector's row (height) axis, so the effective center of rotation
    drifts linearly with row index. This is distinct from an in-plane
    rotation of each projection frame; calibration.find_tilt() detects
    axis lean specifically.

    `tilt_deg` is expressed as `arctan(pixels of center-of-rotation
    drift per row)`. Each row's shift is
    `cor_deviation + tan(tilt_deg) * (row - mid_row)`, which reduces to
    a plain uniform `cor_deviation` shift when `tilt_deg == 0.0`.
    """
    n_angles, height, width = projections.shape
    mid_row = (height - 1) / 2.0
    slope = np.tan(np.radians(tilt_deg))
    shift_per_row = cor_deviation + slope * (np.arange(height) - mid_row)

    out = np.empty_like(projections)
    for row, shift_amount in enumerate(shift_per_row):
        if shift_amount == 0.0:
            out[:, row, :] = projections[:, row, :]
        else:
            out[:, row, :] = ndi.shift(projections[:, row, :], shift=(0, shift_amount), order=1, mode="nearest")
    return out.astype(np.float32)


def make_detector_patterns(
    shape: tuple[int, int],
    inhomogeneity_strength: float,
    n_scratches: int,
    scratch_strength: float,
    n_defective_pixels: int,
    dark_level: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The fixed detector response pattern, shared by every
    projection/flat/dark frame. Returns (gain, dark_offset), both shape
    `shape` (height, width).

    gain: nominally 1.0, with smooth low-frequency non-uniformity, a
    handful of thin vertical scratches (fixed detector columns spanning
    the full height), and a few dead (gain=0) / hot (gain large)
    pixels.

    dark_offset: a uniform baseline.
    """
    height, width = shape

    gain = 1.0 + inhomogeneity_strength * ndi.gaussian_filter(rng.standard_normal(shape), sigma=max(width, height) / 8)

    columns = np.arange(width)
    for _ in range(n_scratches):
        center = rng.uniform(0, width)
        dip = scratch_strength * np.exp(-0.5 * ((columns - center) / 1.5) ** 2)
        gain *= (1.0 - dip)[np.newaxis, :]

    if n_defective_pixels:
        rows = rng.integers(0, height, size=n_defective_pixels)
        cols = rng.integers(0, width, size=n_defective_pixels)
        dead = np.arange(n_defective_pixels) < n_defective_pixels // 2
        gain[rows[dead], cols[dead]] = 0.0
        gain[rows[~dead], cols[~dead]] = 5.0

    gain = np.clip(gain, 0.0, None).astype(np.float32)
    dark_offset = np.full(shape, dark_level, dtype=np.float32)
    return gain, dark_offset


def synthesize_raw(
    line_integrals: np.ndarray,
    gain: np.ndarray,
    dark_offset: np.ndarray,
    flat_intensity: float,
    attenuation_scale: float,
    n_flats: int,
    n_darks: int,
    noise_level: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Turn ideal line integrals plus the detector pattern into raw
    projections, flat frames, and dark frames; see the module docstring
    for the formula. `noise_level` is a relative Gaussian noise
    magnitude applied independently per frame; 0.0 (default) gives
    deterministic, noiseless output.
    """
    def _noise(arr: np.ndarray) -> np.ndarray:
        if noise_level <= 0.0:
            return arr
        return arr + rng.normal(0.0, noise_level * flat_intensity, size=arr.shape)

    transmission = np.exp(-line_integrals * attenuation_scale)
    raw = dark_offset[np.newaxis] + gain[np.newaxis] * flat_intensity * transmission
    raw = _noise(raw).astype(np.float32)

    flats = np.stack([_noise(dark_offset + gain * flat_intensity) for _ in range(n_flats)]).astype(np.float32)
    darks = np.stack([_noise(dark_offset.copy()) for _ in range(n_darks)]).astype(np.float32)

    return raw, flats, darks
