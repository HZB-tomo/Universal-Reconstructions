"""
GPU filtered backprojection (FBP) and backproject-then-filter (BPF) for
algotom_fbp / algotom_bpf, one slice per kernel launch.

Why one slice per launch: algotom's single-slice kernel keeps each
pixel's running sum in a register and writes the pixel once, with
neighbouring threads writing neighbouring memory. Its "chunk" kernel
(many slices in one launch) instead adds into global GPU memory on every
angle for every slice, in a layout that skips around -- far more memory
traffic per update, so a batched call is slower, not faster. Measured
cost of the extra launches is small next to that.

What is overlapped instead: the CPU work around the kernel. FBP's ramp
filter for the next slices runs on worker threads (numpy's FFT releases
the GIL) while the GPU backprojects the current one; BPF's image filter
runs on worker threads after each kernel. The GPU is never left waiting
for a filter that could have been computed ahead.

Mirrors algotom's fbp_reconstruction()/bpf_reconstruction() around
`back_projection_gpu`, including options (`filter_name`, `pad`,
`pad_mode`, `ratio`, `apply_log`, `ramp_win`). Does not modify the
caller's array. Import lazily: needs algotom (and, to run, numba.cuda).
"""
from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from reconstruction.backends._fbp_cpu import attenuation, settings


def _workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def fbp_stack_gpu(rec, sinograms: np.ndarray, centers: np.ndarray, angles_rad: np.ndarray, **options) -> np.ndarray:
    """FBP of a (n_angles, n_slices, width) stack on the GPU, one center per
    slice. Returns (n_slices, width, width) float32."""
    n_angles, n_slices, ncol = sinograms.shape
    s = settings(rec, n_angles, ncol, centers, options, circular=False)
    angles32 = np.float32(angles_rad)
    out = np.empty((n_slices, ncol, ncol), dtype=np.float32)
    scale = np.float32(np.pi / (n_angles - 1))

    def filtered(i: int) -> np.ndarray:
        sino = attenuation(sinograms[:, i, :], s.apply_log)
        return np.ascontiguousarray(
            rec.apply_ramp_filter(sino, s.ramp_win, s.filter_name, s.pad, s.pad_mode), dtype=np.float32,
        )

    workers = _workers()
    window = 2 * workers  # filtered sinograms held ahead of the GPU: bounded memory
    pending: deque = deque()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        submitted = 0
        for i in range(n_slices):
            while submitted < n_slices and len(pending) < window:
                pending.append(pool.submit(filtered, submitted))
                submitted += 1
            sino = pending.popleft().result()
            image = rec.back_projection_gpu(sino, angles32, np.float32(centers[i]))
            if s.mask is not None:
                image = image * s.mask
            out[i] = image * scale
    return out


def bpf_stack_gpu(rec, sinograms: np.ndarray, centers: np.ndarray, angles_rad: np.ndarray, **options) -> np.ndarray:
    """BPF of a (n_angles, n_slices, width) stack on the GPU, one center per
    slice. Returns (n_slices, width, width) float32."""
    n_angles, n_slices, ncol = sinograms.shape
    s = settings(rec, n_angles, ncol, centers, options, circular=True)
    angles32 = np.float32(angles_rad)
    out = np.empty((n_slices, ncol, ncol), dtype=np.float32)
    scale = np.float32(np.pi / (n_angles - 1))

    def finish(i: int, image: np.ndarray) -> None:
        image = rec.apply_circular_ramp_filter(image, s.ramp_win, pad=s.pad, pad_mode=s.pad_mode)
        if s.mask is not None:
            image = image * s.mask
        out[i] = image * scale

    workers = _workers()
    pending: deque = deque()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i in range(n_slices):
            sino = np.ascontiguousarray(attenuation(sinograms[:, i, :], s.apply_log))
            image = rec.back_projection_gpu(sino, angles32, np.float32(centers[i]), edge_pad=True)
            pending.append(pool.submit(finish, i, image))
            while len(pending) > 2 * workers:  # bounded: images waiting for their filter
                pending.popleft().result()
        for future in pending:
            future.result()
    return out
