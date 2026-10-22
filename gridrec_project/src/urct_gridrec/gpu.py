"""
GPU gridrec.

The FFTs (batched 1D, then one 2D) are cuFFT through CuPy. The gridding is
a numba.cuda kernel with one thread per polar sample, adding the sample's
small footprint into the grid with atomic adds. Atomics are needed because
many samples near the grid centre land in the same cells; they are fast
for 32-bit floats, so the grid is kept as separate real and imaginary
float32 arrays (there is no complex atomic add).

The same code runs without a GPU on numba's CUDA simulator
(NUMBA_ENABLE_CUDASIM=1, arrays stay NumPy): slow, but it executes the real
kernel, which is how the logic is tested here. Speed has to be measured on
a real GPU (benchmarks/bench_projects.py --device gpu).
"""
from __future__ import annotations

import math

import numba
import numpy as np
from numba import cuda

from urct_gridrec.core import DefaultOps, Plan

_BLOCK = (32, 8)   # threads: (frequency bins, angles)


@cuda.jit
def _scatter_kernel(spec, cos_t, sin_t, table, width, resolution, origin, grid_re, grid_im):  # pragma: no cover
    kk, j = cuda.grid(2)
    n_angles = spec.shape[0]
    n = spec.shape[1]
    if j >= n_angles or kk >= n:
        return
    val = spec[j, kk]
    vr = val.real
    vi = val.imag
    if vr == 0.0 and vi == 0.0:
        return
    k = kk - n // 2
    gx = k * cos_t[j]
    gy = -k * sin_t[j]
    half_w = width / 2.0
    xl = int(math.floor(gx - half_w)) + 1
    yl = int(math.floor(gy - half_w)) + 1
    for a in range(width):
        wy = table[int((yl + a - gy + half_w) * resolution + 0.5)]
        row = yl + a + origin
        for b in range(width):
            w = wy * table[int((xl + b - gx + half_w) * resolution + 0.5)]
            col = xl + b + origin
            cuda.atomic.add(grid_re, (row, col), vr * w)
            cuda.atomic.add(grid_im, (row, col), vi * w)


def simulated() -> bool:
    return bool(numba.config.ENABLE_CUDASIM)


def gpu_available() -> bool:
    if simulated():
        return True
    try:
        import cupy

        return bool(cuda.is_available()) and cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


class GpuBackend(DefaultOps):
    def __init__(self):
        if simulated():
            self.xp = np
            self._cp = None
        else:
            import cupy

            self.xp = cupy
            self._cp = cupy
        self._table_cache = {}

    def rfft(self, padded_real: np.ndarray):
        xp = self.xp
        return xp.fft.rfft(xp.asarray(padded_real, dtype=xp.float32), axis=1)

    def spectrum(self, plan: Plan, pa, pb, has_b, ca, cb, cos_t, sin_t, scale):
        xp, n = self.xp, plan.n
        half = n // 2
        k = xp.arange(n, dtype=xp.float64) - half

        def full(p):   # bins k = -n/2 .. n/2 - 1 from the non-negative half
            return xp.concatenate([xp.conj(p[:, half:0:-1]), p[:, :half]], axis=1)

        filt = xp.asarray(np.concatenate([plan.filt[half:0:-1], plan.filt[:half]]))
        shift = xp.asarray(plan.delta * (cos_t - sin_t))

        def phase(center):
            return xp.exp(2j * np.pi * k[None, :] * (center + shift)[:, None] / n)

        s = full(pa) * phase(ca)
        if has_b:
            s = s + 1j * full(pb) * phase(cb)
        return (s * filt[None, :] * scale).astype(xp.complex64)

    def scatter(self, plan: Plan, spec, cos_t, sin_t):
        xp = self.xp
        kern = plan.kernel
        key = id(plan)
        if key not in self._table_cache:
            self._table_cache[key] = xp.asarray(kern.table)
        n_angles, n = spec.shape
        grid_re = xp.zeros((plan.padded, plan.padded), dtype=xp.float32)
        grid_im = xp.zeros((plan.padded, plan.padded), dtype=xp.float32)
        blocks = ((n + _BLOCK[0] - 1) // _BLOCK[0], (n_angles + _BLOCK[1] - 1) // _BLOCK[1])
        _scatter_kernel[blocks, _BLOCK](
            spec, xp.asarray(cos_t), xp.asarray(sin_t), self._table_cache[key],
            kern.width, kern.resolution, plan.origin, grid_re, grid_im,
        )
        return grid_re + 1j * grid_im

    def ifft2(self, core):
        return self.xp.fft.ifft2(core)

    def to_host(self, a) -> np.ndarray:
        return self._cp.asnumpy(a) if self._cp is not None else np.asarray(a)
