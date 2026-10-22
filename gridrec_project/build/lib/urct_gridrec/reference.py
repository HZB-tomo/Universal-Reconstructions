"""
Slow, simple reference implementations used to check the fast ones.

ReferenceBackend runs the same algorithm in plain NumPy float64 (gridding
by np.add.at, no threads). direct_reconstruction() skips gridding entirely:
it evaluates the Fourier sum at every pixel exactly, so it measures the
gridding error itself. Both are for small images only.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from urct_gridrec.core import DefaultOps, Plan


class ReferenceBackend(DefaultOps):
    xp = np

    def prepare(self, plan: Plan, sino, index, apply_log):
        return super().prepare(plan, sino, index, apply_log).astype(np.float64)

    def rfft(self, padded_real: np.ndarray) -> np.ndarray:
        return np.fft.rfft(padded_real.astype(np.float64), axis=1)

    def spectrum(self, plan: Plan, pa, pb, has_b, ca, cb, cos_t, sin_t, scale) -> np.ndarray:
        n = plan.n
        half = n // 2
        k = np.arange(n) - half

        def full(p):   # bins k = -n/2 .. n/2 - 1 from the non-negative half (real input -> Hermitian)
            return np.concatenate([np.conj(p[:, half:0:-1]), p[:, :half]], axis=1)

        filt = np.concatenate([plan.filt[half:0:-1], plan.filt[:half]])
        shift = plan.delta * (cos_t - sin_t)

        def phase(center):
            return np.exp(2j * np.pi * k[None, :] * (center + shift)[:, None] / n)

        s = full(pa) * phase(ca)
        if has_b:
            s = s + 1j * full(pb) * phase(cb)
        return s * filt[None, :] * scale

    def scatter(self, plan: Plan, spec, cos_t, sin_t) -> np.ndarray:
        kern = plan.kernel
        n, half_w, res = plan.n, kern.width / 2.0, kern.resolution
        table = kern.table.astype(np.float64)
        grid = np.zeros((plan.padded, plan.padded), dtype=np.complex128)
        k = np.arange(n) - n // 2
        for j in range(spec.shape[0]):
            gx, gy = k * cos_t[j], -k * sin_t[j]
            xl = np.floor(gx - half_w).astype(int) + 1
            yl = np.floor(gy - half_w).astype(int) + 1
            for a in range(kern.width):
                wy = table[((yl + a - gy + half_w) * res + 0.5).astype(int)]
                for b in range(kern.width):
                    wx = table[((xl + b - gx + half_w) * res + 0.5).astype(int)]
                    np.add.at(grid, (yl + a + plan.origin, xl + b + plan.origin), spec[j] * wy * wx)
        return grid

    def ifft2(self, core: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(core)

    @staticmethod
    def to_host(a: np.ndarray) -> np.ndarray:
        return np.asarray(a)


def direct_reconstruction(atten: np.ndarray, angles_rad: np.ndarray, center: float, plan: Plan) -> np.ndarray:
    """
    One slice, (angles, width) attenuation sinogram -> (width, width), by the
    exact Fourier sum over every sample at every pixel: no gridding, no
    oversampling error. The same spectrum, filter and scaling as gridrec.
    """
    from urct_gridrec.core import _pad

    n, width = plan.n, plan.width
    backend = ReferenceBackend()
    pa = backend.rfft(_pad(plan, atten))
    no_shift = replace(plan, delta=0.0)       # pixel positions are evaluated exactly, no half-pixel trick
    cos_t, sin_t = np.cos(angles_rad), np.sin(angles_rad)
    scale = float(np.pi / (len(angles_rad) - 1))
    spec = backend.spectrum(no_shift, pa, pa, False, center + plan.off, 0.0, cos_t, sin_t, scale)
    k = (np.arange(n) - n // 2) / n
    pos = np.arange(width) - (width - 1) / 2.0           # pixel offsets from the image center
    out = np.zeros((width, width))
    for j in range(len(angles_rad)):
        # exp(2 pi i k/n (x cos - y sin)) separates into an x factor times a y factor
        ex = np.exp(2j * np.pi * np.outer(pos * cos_t[j], k))     # (x, k)
        ey = np.exp(2j * np.pi * np.outer(-pos * sin_t[j], k))    # (y, k)
        out += ((ey * spec[j][None, :]) @ ex.T).real
    return out / (n * n)
