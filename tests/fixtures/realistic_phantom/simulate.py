"""
Turns a phantom volume (phantom.py) into realistic, detector-level raw
data: parallel-beam forward projection, center-of-rotation deviation,
detector tilt, and a simulated flat/dark-imperfect detector.

The forward model is deliberately the exact inverse of
preprocessing.normalize()'s formula, so that running normalize() on
this generator's output (with the flat/dark it also produces) recovers
the ideal transmission almost exactly -- a real quantitative check on
the rest of the pipeline, not just a plausible-looking test file:

    transmission = exp(-line_integral * attenuation_scale)      (Beer-Lambert)
    raw          = dark_offset + gain * flat_intensity * transmission + noise
    flat_frame   = dark_offset + gain * flat_intensity + noise
    dark_frame   = dark_offset + noise

`gain` (spatial non-uniformity, scratches, dead/hot pixels) and
`dark_offset` are the TRUE, fixed detector patterns -- shared between
every projection, flat, and dark frame, exactly like a real detector's
defects are a property of the hardware, not of one exposure. This is
also why ring artifacts need no separate simulation step: they fall
directly out of the same fixed-pattern `gain` map being present, and
imperfectly corrected, in every projection angle.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi


def project_parallel(volume: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """
    Ideal, noise-free parallel-beam line integrals: rotate the volume
    around its height axis (axes 1/2, the square cross-section) by each
    angle and sum along one in-plane axis. Returns shape
    (n_angles, height, width) -- the same convention used everywhere
    else in this codebase.

    The rotation sign (negated) and summed axis (1, not 2) were both
    verified empirically against algotom.util.simulation.make_sinogram()
    (a Fourier-slice-theorem sinogram generator, independent of this
    rotate-and-sum approach) as a known-good reference -- the original
    version of this function (angle un-negated, summed along axis 2)
    only reached ~0.61 correlation against that reference; this version
    reaches ~0.97, matching to within the interpolation error expected
    from a rotate-based (rather than Fourier-based) projector. That
    verification was development-time only (via a throwaway script, not
    a project dependency) -- algotom stays an optional dependency of
    this project overall, and is NOT required to run this function.
    """
    projections = np.empty((len(angles_deg), *volume.shape[:2]), dtype=np.float32)
    for i, angle in enumerate(angles_deg):
        rotated = ndi.rotate(volume, -angle, axes=(1, 2), reshape=False, order=1, mode="constant", cval=0.0)
        projections[i] = rotated.sum(axis=1)
    return projections


def apply_cor_and_tilt(projections: np.ndarray, cor_deviation: float, tilt_deg: float) -> np.ndarray:
    """
    Inject a known center-of-rotation deviation (sub-pixel lateral
    shift, every projection) and detector tilt (in-plane rotation of
    every projection image) -- the forward-direction equivalent of the
    tilt-correction step in oldscripts (which rotated projections back
    to remove tilt; this rotates them to introduce it).
    """
    out = projections
    if tilt_deg:
        out = np.stack([ndi.rotate(p, tilt_deg, reshape=False, order=1, mode="nearest") for p in out])
    if cor_deviation:
        out = ndi.shift(out, shift=(0, 0, cor_deviation), order=1, mode="nearest")
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
    The TRUE, fixed detector response pattern -- shared by every
    projection/flat/dark frame. Returns (gain, dark_offset), both shape
    `shape` (height, width), matching one detector frame.

    gain: nominally 1.0, with smooth low-frequency non-uniformity
    (detector element gain variation), a handful of thin vertical
    scratches (fixed detector columns, spanning the full height -- the
    kind of defect that produces a ring at a fixed radius once
    imperfectly corrected), and a few dead (gain=0) / hot (gain large)
    pixels.

    dark_offset: a uniform baseline -- no defects requested for dark,
    unlike flat.
    """
    height, width = shape

    # Smooth non-uniformity: white noise low-pass filtered.
    gain = 1.0 + inhomogeneity_strength * ndi.gaussian_filter(rng.standard_normal(shape), sigma=max(width, height) / 8)

    # Scratches: thin vertical lines (fixed columns, full height) with a
    # Gaussian-profile dip in gain.
    columns = np.arange(width)
    for _ in range(n_scratches):
        center = rng.uniform(0, width)
        dip = scratch_strength * np.exp(-0.5 * ((columns - center) / 1.5) ** 2)
        gain *= (1.0 - dip)[np.newaxis, :]

    # Defective pixels: half dead (gain ~0), half stuck-high.
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
    Turn ideal line integrals plus the true detector pattern into raw
    projections, flat frames, and dark frames -- see the module
    docstring for the exact formula. `noise_level` is a relative
    Gaussian noise magnitude applied independently per frame; 0.0
    (the default) gives fully deterministic, noiseless output, which is
    the right default for testing exact numerical agreement -- turn it
    up for something closer to real acquisition noise.
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
