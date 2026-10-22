"""
Race-free CPU filtered backprojection for algotom_fbp.

algotom's own back_projection_cpu() is parallelized over the ANGLES
(`prange(sino_height)`), but every angle adds into the same output image
(`recon[y, x] += val`), so threads overwrite each other's updates: the
result is different on every run and about 4-5% off on a typical
256x256 slice (measured: two identical runs differ by ~4%, and the sum
of the image is ~4% low versus a single-threaded run).

This kernel parallelizes over output ROWS instead: each thread owns its
rows exclusively and adds the angles in a fixed order, so the result is
exactly reproducible and equals a single-threaded run.

numba is imported here, so import this module lazily (the backend does).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def back_projection(sinogram: np.ndarray, angles: np.ndarray, center: float, edge_pad: bool = False) -> np.ndarray:  # pragma: no cover
    """Backproject a (filtered) float32 sinogram (n_angles, width) onto a
    (width, width) float32 image. Same geometry and linear interpolation
    as algotom's back_projection_cpu(). `edge_pad` extends the sinogram's
    first/last column over rays falling outside the detector (BPF only)."""
    n_angles, width = sinogram.shape
    last = width - 1
    icenter = 0.5 * last
    cos_t = np.empty(n_angles)
    sin_t = np.empty(n_angles)
    for i in range(n_angles):
        theta = -np.float64(angles[i])
        cos_t[i] = np.cos(theta)
        sin_t[i] = np.sin(theta)

    recon = np.zeros((width, width), dtype=np.float32)
    for y in prange(width):
        y_cor = y - icenter
        for i in range(n_angles):
            c = cos_t[i]
            s = sin_t[i]
            y_term = y_cor * s + center
            for x in range(width):
                f_pos = (x - icenter) * c + y_term
                if 0.0 <= f_pos <= last:
                    d_pos = np.int32(np.floor(f_pos))
                    u_pos = np.int32(np.ceil(f_pos))
                    if u_pos != d_pos:
                        yd = sinogram[i, d_pos]
                        yu = sinogram[i, u_pos]
                        recon[y, x] += yd + (yu - yd) * (f_pos - d_pos)
                    else:
                        recon[y, x] += sinogram[i, d_pos]
                elif edge_pad:
                    if f_pos < 0.0:
                        recon[y, x] += sinogram[i, 0]
                    else:
                        recon[y, x] += sinogram[i, last]
    return recon


@dataclass
class Settings:
    """The reconstruction options shared by every FBP/BPF implementation
    (CPU and GPU), resolved once per call the way algotom does."""

    filter_name: str
    pad: int
    pad_mode: str
    apply_log: bool
    ramp_win: np.ndarray
    mask: Optional[np.ndarray]


def settings(rec, n_angles: int, ncol: int, centers: np.ndarray, options: dict, circular: bool) -> Settings:
    """`circular` selects BPF's image-domain ramp window instead of FBP's sinogram one."""
    from algotom.util import utility as util

    filter_name = options.get("filter_name", "hann")
    pad = options.get("pad")
    pad = min(int(0.15 * ncol), 150) if pad is None else pad
    ramp_win = options.get("ramp_win")
    if ramp_win is None:
        ramp_win = (
            rec.make_circular_ramp_window(ncol + 2 * pad, filter_name) if circular
            else rec.make_2d_ramp_window(n_angles, ncol + 2 * pad, filter_name)
        )
    ratio = options.get("ratio", 1.0)
    mask = None
    if ratio is not None:
        mask = util.make_circle_mask(ncol, min(centers[0], ncol - centers[0]) / (0.5 * ncol) if ratio == 0.0 else ratio)
    return Settings(
        filter_name=filter_name, pad=pad, pad_mode=options.get("pad_mode", "edge"),
        apply_log=options.get("apply_log", True), ramp_win=ramp_win, mask=mask,
    )


def attenuation(sino_column: np.ndarray, apply_log: bool) -> np.ndarray:
    """One slice's sinogram as a float32 copy (the caller's array is never touched),
    with -log applied the way algotom does: values <= 0 become 1 first."""
    sino = np.array(sino_column, dtype=np.float32)
    if apply_log:
        sino[sino <= 0.0] = np.float32(1.0)
        sino = -np.log(sino)
    return sino


def fbp_stack(rec, sinograms: np.ndarray, centers: np.ndarray, angles_rad: np.ndarray, **options) -> np.ndarray:
    """
    FBP of a (n_angles, n_slices, width) stack on the CPU, one center per
    slice, using algotom's own ramp filtering and options (`filter_name`,
    `pad`, `pad_mode`, `ratio`, `apply_log`, `ramp_win`) around the
    race-free back_projection() above. Does not modify `sinograms`.
    Returns (n_slices, width, width) float32.
    """
    n_angles, n_slices, ncol = sinograms.shape
    s = settings(rec, n_angles, ncol, centers, options, circular=False)
    angles32 = np.float32(angles_rad)
    out = np.empty((n_slices, ncol, ncol), dtype=np.float32)
    for i in range(n_slices):
        sino = attenuation(sinograms[:, i, :], s.apply_log)
        filtered = np.ascontiguousarray(
            rec.apply_ramp_filter(sino, s.ramp_win, s.filter_name, s.pad, s.pad_mode), dtype=np.float32,
        )
        image = back_projection(filtered, angles32, np.float32(centers[i]))
        if s.mask is not None:
            image = image * s.mask
        out[i] = image * np.float32(np.pi / (n_angles - 1))
    return out


def bpf_stack(rec, sinograms: np.ndarray, centers: np.ndarray, angles_rad: np.ndarray, **options) -> np.ndarray:
    """
    Backproject-then-filter (BPF) of a (n_angles, n_slices, width) stack on
    the CPU, one center per slice: edge-padded backprojection of the
    unfiltered sinogram, then algotom's circular ramp filter on the image.
    Mirrors algotom's bpf_reconstruction() around the race-free
    back_projection(). Does not modify `sinograms`.
    Returns (n_slices, width, width) float32.
    """
    n_angles, n_slices, ncol = sinograms.shape
    s = settings(rec, n_angles, ncol, centers, options, circular=True)
    angles32 = np.float32(angles_rad)
    out = np.empty((n_slices, ncol, ncol), dtype=np.float32)
    for i in range(n_slices):
        sino = attenuation(sinograms[:, i, :], s.apply_log)
        image = back_projection(np.ascontiguousarray(sino), angles32, np.float32(centers[i]), True)
        image = rec.apply_circular_ramp_filter(image, s.ramp_win, pad=s.pad, pad_mode=s.pad_mode)
        if s.mask is not None:
            image = image * s.mask
        out[i] = image * np.float32(np.pi / (n_angles - 1))
    return out
