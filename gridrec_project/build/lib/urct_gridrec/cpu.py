"""
CPU gridrec with Numba.

The expensive step is gridding: every polar sample adds a small kernel
footprint into a shared grid. Done naively in parallel that is a data race
(two threads adding into the same cell lose updates -- the same bug that
makes algotom's CPU backprojection wrong). Here the grid is split into
bands of rows and each thread only ever writes the rows of its own bands,
so there are no races, the result is exactly reproducible, and no atomics
or per-thread grids are needed. For each band and angle the range of
samples that can reach the band is computed directly, so a thread visits
(almost) only its own work.
"""
from __future__ import annotations

import numba
import numpy as np
import scipy.fft as sfft
from numba import njit, prange

from urct_gridrec.core import DefaultOps, Plan, crop_pieces

_BAND_ROWS = 16


@njit(parallel=True, cache=True)
def _prepare(sino, index, off, n, apply_log):  # pragma: no cover
    """Slice `index` of the (angles, slices, width) stack, optionally -log (values <= 0
    count as 1, like FBP), edge-padded to length n: one parallel pass, no copy of the stack."""
    n_angles, _, width = sino.shape
    out = np.empty((n_angles, n), dtype=np.float32)
    for j in prange(n_angles):
        for t in range(width):
            v = sino[j, index, t]
            if apply_log:
                if v <= 0.0:
                    v = np.float32(1.0)
                v = -np.log(v)
            out[j, off + t] = v
        left = out[j, off]
        right = out[j, off + width - 1]
        for t in range(off):
            out[j, t] = left
        for t in range(off + width, n):
            out[j, t] = right
    return out


@njit(parallel=True, cache=True)
def _spectrum(pa, pb, has_b, ca, cb, delta, cos_t, sin_t, filt, scale, n):  # pragma: no cover
    """Samples S[j, k] (k = -n/2 .. n/2 - 1) of the packed spectrum Sa + i Sb. Each real
    slice's spectrum is Hermitian, so only k >= 0 is computed, and the phase factors
    exp(i w k) come from a running product instead of a sine and cosine per sample."""
    n_angles = pa.shape[0]
    half = n // 2
    spec = np.empty((n_angles, n), dtype=np.complex64)
    two_pi = 2.0 * np.pi
    for j in prange(n_angles):
        shift = delta * (cos_t[j] - sin_t[j])
        wa = two_pi * (ca + shift) / n
        wb = two_pi * (cb + shift) / n
        step_a = complex(np.cos(wa), np.sin(wa))
        step_b = complex(np.cos(wb), np.sin(wb))
        ph_a = complex(1.0, 0.0)
        ph_b = complex(1.0, 0.0)
        for m in range(half + 1):
            f = filt[m] * scale
            sa = np.complex128(pa[j, m]) * ph_a
            sb = np.complex128(pb[j, m]) * ph_b if has_b else complex(0.0, 0.0)
            if m < half:
                spec[j, half + m] = (sa + 1j * sb) * f
            if m >= 1:
                spec[j, half - m] = (np.conj(sa) + 1j * np.conj(sb)) * f
            ph_a *= step_a
            ph_b *= step_b
    return spec


@njit(parallel=True, cache=True)
def _scatter(spec, cos_t, sin_t, table, width, resolution, origin, padded, n_threads):  # pragma: no cover
    n_angles, n = spec.shape
    half_n = n // 2
    half_w = width / 2.0
    grid = np.zeros((padded, padded), dtype=np.complex64)
    n_bands = (padded + _BAND_ROWS - 1) // _BAND_ROWS
    for t in prange(n_threads):
        wx = np.empty(width, dtype=np.float32)
        for band in range(t, n_bands, n_threads):     # interleaved: the dense centre rows are shared out
            r0 = band * _BAND_ROWS
            r1 = min(r0 + _BAND_ROWS, padded)
            g_min = r0 - origin - half_w - 1.0        # samples whose footprint can touch rows [r0, r1)
            g_max = r1 - 1 - origin + half_w + 1.0
            for j in range(n_angles):
                c = cos_t[j]
                s = sin_t[j]                          # grid y of sample k is -k * s
                if abs(s) < 1e-12:
                    if g_min <= 0.0 and 0.0 <= g_max:
                        k_lo = -half_n
                        k_hi = half_n - 1
                    else:
                        continue
                else:
                    k1 = -g_max / s
                    k2 = -g_min / s
                    k_lo = max(-half_n, int(np.ceil(min(k1, k2))))
                    k_hi = min(half_n - 1, int(np.floor(max(k1, k2))))
                for k in range(k_lo, k_hi + 1):
                    val = spec[j, k + half_n]
                    gx = k * c
                    gy = -k * s
                    xl = int(np.floor(gx - half_w)) + 1
                    yl = int(np.floor(gy - half_w)) + 1
                    a0 = max(0, r0 - (yl + origin))
                    a1 = min(width, r1 - (yl + origin))
                    if a0 >= a1:
                        continue
                    for b in range(width):
                        wx[b] = table[int((xl + b - gx + half_w) * resolution + 0.5)]
                    for a in range(a0, a1):
                        wy = table[int((yl + a - gy + half_w) * resolution + 0.5)]
                        vy = val * wy
                        row = yl + a + origin
                        col = xl + origin
                        for b in range(width):
                            grid[row, col + b] += vy * wx[b]
    return grid


@njit(parallel=True, cache=True)
def _fold_shift(grid, out_rows, out_cols, origin, n):  # pragma: no cover
    """Fold the padded grid's margins back into the periodic n x n grid and write it in FFT
    order (coordinate 0 first), in one pass. out_rows / out_cols[i] list the (up to 3) padded
    rows / columns that land on output row / column i, -1 for none."""
    out = np.zeros((n, n), dtype=np.complex64)
    for r in prange(n):
        for ry in range(3):
            py = out_rows[r, ry]
            if py < 0:
                continue
            for c in range(n):
                acc = out[r, c]
                for cx in range(3):
                    px = out_cols[c, cx]
                    if px >= 0:
                        acc += grid[py, px]
                out[r, c] = acc
    return out


@njit(parallel=True, cache=True)
def _finish(image, inv_deapod, mask, out_a, out_b, has_b):  # pragma: no cover
    n_rows, n_cols = image.shape
    for y in prange(n_rows):
        for x in range(n_cols):
            v = image[y, x]
            d = inv_deapod[y, x] * mask[y, x]
            out_a[y, x] = v.real * d
            if has_b:
                out_b[y, x] = v.imag * d


def _fold_tables(n: int, origin: int, padded: int) -> np.ndarray:
    """For each FFT-order index i of an axis of length n (position i if i < n/2 else i - n),
    the padded-array positions p = idx + shift * n + origin (shift -1, 0, 1) that are in range."""
    idx = np.where(np.arange(n) < n // 2, np.arange(n), np.arange(n) - n)
    table = np.full((n, 3), -1, dtype=np.int64)
    for slot, shift in enumerate((-1, 0, 1)):
        p = idx + shift * n + origin
        table[:, slot] = np.where((p >= 0) & (p < padded), p, -1)
    return table


class CpuBackend(DefaultOps):
    """NumPy/SciPy FFTs on all cores; Numba for the spectrum, the gridding and the passes around them."""

    xp = np

    def __init__(self):
        self._tables = {}

    def prepare(self, plan: Plan, sino: np.ndarray, index: int, apply_log: bool) -> np.ndarray:
        return _prepare(sino, index, plan.off, plan.n, apply_log)

    def rfft(self, padded_real: np.ndarray) -> np.ndarray:
        return sfft.rfft(padded_real.astype(np.float32, copy=False), axis=1, workers=-1)

    def spectrum(self, plan: Plan, pa, pb, has_b, ca, cb, cos_t, sin_t, scale) -> np.ndarray:
        return _spectrum(pa, pb, has_b, ca, cb, plan.delta, cos_t, sin_t, plan.filt, scale, plan.n)

    def scatter(self, plan: Plan, spec, cos_t, sin_t) -> np.ndarray:
        k = plan.kernel
        return _scatter(spec, cos_t, sin_t, k.table, k.width, k.resolution, plan.origin, plan.padded,
                        numba.get_num_threads())

    def fold_shift(self, plan: Plan, grid) -> np.ndarray:
        key = (plan.n, plan.origin, plan.padded)
        if key not in self._tables:
            self._tables[key] = _fold_tables(plan.n, plan.origin, plan.padded)
        table = self._tables[key]
        return _fold_shift(grid, table, table, plan.origin, plan.n)

    def ifft2(self, core: np.ndarray) -> np.ndarray:
        return sfft.ifft2(core, workers=-1)

    def ifft2_crop(self, plan: Plan, core: np.ndarray) -> np.ndarray:
        """Only width x width of the n x n image is wanted, so transform along the columns
        first, keep just the needed columns, and then transform those along the rows:
        (n + width) / (2 n) of the work of a full 2D transform, and no big copies."""
        pieces = crop_pieces(plan.n, plan.width)
        t = sfft.ifft(core, axis=1, workers=-1)
        t = np.concatenate([t[:, p] for p in pieces], axis=1)                 # (n, width)
        t = sfft.ifft(t, axis=0, workers=-1)
        return np.concatenate([t[p] for p in pieces], axis=0)                 # (width, width)

    def finish(self, image, inv_deapod, mask, out_a, out_b) -> None:
        has_b = out_b is not None
        _finish(image, inv_deapod, mask, out_a, out_b if has_b else out_a, has_b)

    @staticmethod
    def to_host(a: np.ndarray) -> np.ndarray:
        return np.asarray(a)
