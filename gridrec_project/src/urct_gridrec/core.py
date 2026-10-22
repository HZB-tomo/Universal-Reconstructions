"""
The gridrec algorithm, independent of where it runs.

Gridrec (Dowd, De Carlo, Rivers, Jacobsen et al., J. Synchrotron Rad. 6,
255, 1999) is filtered backprojection done in the Fourier domain. By the
Fourier slice theorem the 1D FFT of a projection is a line of the image's
2D spectrum, so for each pair of slices:

  1. pad the sinograms to length N (edge values), 1D FFT every projection;
  2. multiply by the ramp (times a smoothing window) and by a phase that
     moves the origin to the rotation axis;
  3. "grid": spread every polar sample (angle, k) onto a Cartesian grid
     with a small kernel (Kaiser-Bessel);
  4. one 2D inverse FFT of the grid;
  5. divide out the kernel's Fourier transform, crop to the image.

Cost per slice is about width^2 log(width) instead of backprojection's
width^2 x angles. Two real slices share one complex transform (slice a in
the real part, slice b in the imaginary part), which halves the work.

Conventions match the FBP backends of the main project: sinograms are
(angles, slices, width), angles in degrees, the image has shape
(slices, width, width) with axes (y, x), and the result is scaled by
pi / (n_angles - 1) like algotom's FBP. A pixel at column x, row y sees
the detector position  (x - c0) cos(t) - (y - c0) sin(t) + center,
c0 = (width - 1) / 2.

The sample at angle t and signed frequency bin k sits on the grid at
(k cos t, -k sin t) (grid cells = FFT bins, because the padded projection
length N equals the grid size).
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.fft as sfft

from urct_gridrec.filters import filter_response
from urct_gridrec.kernel import Kernel


@dataclass
class Plan:
    """Everything that depends only on the image width and the settings."""

    width: int
    n: int                   # padded projection length == grid size (even)
    off: int                 # where the sinogram starts inside the padded projection
    kernel: Kernel
    margin: int              # extra grid rows/columns around the n x n grid for the kernel footprint
    padded: int              # n + 2 * margin
    origin: int              # array index of grid coordinate 0 in the padded grid
    filt: np.ndarray         # (n // 2 + 1,) ramp x window for k = 0 .. n / 2
    crop: np.ndarray         # (width,) indices of the image pixels in FFT order
    deapod: np.ndarray       # (width, width) kernel correction
    delta: float             # half-pixel shift of the pixel positions (0.5 for even widths)


@lru_cache(maxsize=16)
def make_plan(
    width: int, oversampling: float = 1.25, kernel_width: int = 6,
    filter_name: str = "hann", table_resolution: int = 4096,
) -> Plan:
    """Everything that depends only on the width and the settings. Cached: a Plan is read-only,
    so repeated calls (every chunk of a streamed run, every preview) share one."""
    n = sfft.next_fast_len(int(np.ceil(oversampling * width)))
    while n % 2:
        n = sfft.next_fast_len(n + 1)
    kernel = Kernel.build(kernel_width, n / width, table_resolution)
    margin = kernel_width // 2 + 2
    m = np.arange(width) - width // 2          # pixel n sits at FFT position m = n - width // 2
    ap = kernel.transform(m / n)
    return Plan(
        width=width, n=n, off=(n - width) // 2, kernel=kernel, margin=margin, padded=n + 2 * margin,
        origin=n // 2 + margin, filt=filter_response(filter_name, n), crop=m % n,
        deapod=np.outer(ap, ap), delta=0.5 if width % 2 == 0 else 0.0,
    )


def circle_mask(width: int, ratio: float) -> np.ndarray:
    c = (width - 1) / 2.0
    y, x = np.ogrid[:width, :width]
    return ((x - c) ** 2 + (y - c) ** 2 <= (ratio * width / 2.0) ** 2).astype(np.float32)


def _pad(plan: Plan, sino: np.ndarray) -> np.ndarray:
    return np.pad(sino, ((0, 0), (plan.off, plan.n - plan.width - plan.off)), mode="edge")


def crop_pieces(n: int, width: int) -> list[slice]:
    """Slices of an FFT-ordered axis of length n holding image pixels 0 .. width-1
    (pixel p sits at FFT position (p - width // 2) mod n), in pixel order."""
    below = width // 2           # pixels at negative positions: the end of the axis
    return [slice(n - below, n), slice(0, width - below)]


class DefaultOps:
    """Straightforward NumPy/CuPy versions of the steps around the gridding;
    backends override the ones they can do faster (see cpu.py)."""

    def prepare(self, plan: Plan, sino: np.ndarray, index: int, apply_log: bool):
        """Slice `index` of the raw (angles, slices, width) stack as an attenuation
        sinogram padded to the FFT length, without touching `sino`."""
        x = np.array(sino[:, index, :], dtype=np.float32)
        if apply_log:
            x[x <= 0.0] = 1.0
            x = -np.log(x)
        return _pad(plan, x)

    def fold_shift(self, plan: Plan, grid):
        """Fold the kernel footprint that stuck out of the n x n grid back in (the
        grid is periodic) and move grid coordinate 0 to the FFT origin."""
        n, h = plan.n, plan.margin
        grid[n:n + h, :] += grid[0:h, :]
        grid[h:2 * h, :] += grid[h + n:h + n + h, :]
        grid[:, n:n + h] += grid[:, 0:h]
        grid[:, h:2 * h] += grid[:, h + n:h + n + h]
        return self.xp.fft.ifftshift(grid[h:h + n, h:h + n])

    def finish(self, image, inv_deapod, mask, out_a, out_b):
        """Divide out the kernel's transform, apply the field-of-view mask, and write the real
        part to out_a and (if given) the imaginary part to out_b."""
        image = image * inv_deapod
        out_a[:] = self.to_host(image.real) * mask
        if out_b is not None:
            out_b[:] = self.to_host(image.imag) * mask

    def ifft2_crop(self, plan: Plan, core):
        """Inverse 2D FFT of the n x n spectrum, then the width x width image."""
        xp = self.xp
        image = self.ifft2(core)
        rows = xp.concatenate([image[p] for p in crop_pieces(plan.n, plan.width)], axis=0)
        return xp.concatenate([rows[:, p] for p in crop_pieces(plan.n, plan.width)], axis=1)


@contextmanager
def _stage(timings: dict | None, name: str):
    """Adds the time spent in the block to timings[name] (no-op when timings is None)."""
    if timings is None:
        yield
        return
    t = time.perf_counter()
    yield
    timings[name] = timings.get(name, 0.0) + time.perf_counter() - t


def reconstruct_stack(
    sino: np.ndarray, angles_rad: np.ndarray, centers: np.ndarray, plan: Plan, backend, ratio: float | None = 1.0,
    timings: dict | None = None, apply_log: bool = True,
) -> np.ndarray:
    """
    Gridrec of a stack `sino` (angles, slices, width) of transmission values (or
    attenuation with apply_log=False). `backend` supplies prepare / rfft /
    spectrum / scatter / fold_shift / ifft2_crop (see cpu.py, gpu.py, reference.py).
    `sino` is only read. Returns (slices, width, width) float32.
    """
    n_angles, n_slices, width = sino.shape
    if width != plan.width:
        raise ValueError(f"plan was made for width {plan.width}, got {width}")
    xp = backend.xp
    out = np.empty((n_slices, width, width), dtype=np.float32)
    cos_t, sin_t = np.cos(angles_rad), np.sin(angles_rad)
    scale = float(np.pi / (n_angles - 1))
    inv_deapod = xp.asarray((1.0 / plan.deapod).astype(np.float32))
    mask = xp.asarray(circle_mask(width, ratio)) if ratio is not None else xp.ones((width, width), dtype=xp.float32)

    for a in range(0, n_slices, 2):
        b = a + 1 if a + 1 < n_slices else None
        with _stage(timings, "pad+log"):
            padded_a = backend.prepare(plan, sino, a, apply_log)
            padded_b = backend.prepare(plan, sino, b, apply_log) if b is not None else None
        with _stage(timings, "rfft"):
            pa = backend.rfft(padded_a)
            pb = backend.rfft(padded_b) if b is not None else pa
        with _stage(timings, "spectrum"):
            spec = backend.spectrum(
                plan, pa, pb, b is not None, centers[a] + plan.off,
                (centers[b] if b is not None else 0.0) + plan.off, cos_t, sin_t, scale,
            )
        with _stage(timings, "scatter"):
            grid = backend.scatter(plan, spec, cos_t, sin_t)
        with _stage(timings, "fold+shift"):
            core = backend.fold_shift(plan, grid)
        with _stage(timings, "ifft2+crop"):
            image = backend.ifft2_crop(plan, core)
        with _stage(timings, "deapod+out"):
            backend.finish(image, inv_deapod, mask, out[a], out[b] if b is not None else None)
    return out
