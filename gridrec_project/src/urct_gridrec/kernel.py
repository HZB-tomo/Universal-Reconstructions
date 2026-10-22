"""
Gridding kernel: Kaiser-Bessel window, sampled into a lookup table, plus
its continuous Fourier transform for the "deapodization" correction.

Spreading a sample onto the grid multiplies the image by the kernel's
Fourier transform; dividing by it afterwards undoes that. Kaiser-Bessel
is the standard near-optimal choice for gridding (Beatty et al., IEEE
TMI 2005) and, unlike the prolate spheroidal window classic gridrec
uses, needs no special-function tables beyond numpy's I0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def kb_beta(width: int, oversampling: float) -> float:
    """Shape parameter of the Kaiser-Bessel window for a kernel `width` grid
    cells wide at the given oversampling ratio (Beatty et al. 2005)."""
    return float(np.pi * np.sqrt((width / oversampling) ** 2 * (oversampling - 0.5) ** 2 - 0.8))


@dataclass(frozen=True)
class Kernel:
    width: int            # support in grid cells; nonzero for |d| < width / 2
    resolution: int       # table entries per grid cell
    beta: float
    table: np.ndarray     # float32, table[t] = phi(t / resolution - width / 2), t = 0 .. width * resolution,
                          # plus one repeated end value (so a lookup at the very end stays in range)

    @staticmethod
    def build(width: int = 6, oversampling: float = 2.0, resolution: int = 4096) -> "Kernel":
        beta = kb_beta(width, oversampling)
        d = np.arange(width * resolution + 1) / resolution - width / 2.0
        x = np.clip(2.0 * d / width, -1.0, 1.0)
        table = np.i0(beta * np.sqrt(1.0 - x * x)) / np.i0(beta)
        return Kernel(width, resolution, beta, np.append(table, table[-1]).astype(np.float32))

    def transform(self, s: np.ndarray) -> np.ndarray:
        """Continuous Fourier transform of the kernel at `s` (cycles per grid cell), in
        closed form: L / I0(beta) * sinh(r) / r with r = sqrt(beta^2 - (pi L s)^2), which
        becomes sin(r') / r' beyond the edge pi L s = beta. Real and even."""
        x = np.pi * self.width * np.asarray(s, dtype=np.float64)
        z = self.beta ** 2 - x ** 2
        root = np.sqrt(np.abs(z))
        with np.errstate(invalid="ignore", divide="ignore"):
            shape = np.where(z > 0, np.sinh(root) / root, np.sin(root) / root)
        shape = np.where(root == 0.0, 1.0, shape)
        return shape * self.width / np.i0(self.beta)

    def transform_numeric(self, s: np.ndarray) -> np.ndarray:
        """The same transform by numerical integration of the table: slow, used to check transform()."""
        n = self.width * self.resolution + 1
        d = np.arange(n) / self.resolution - self.width / 2.0
        phi = self.table[:n].astype(np.float64)
        phi[0] *= 0.5   # trapezoid rule end points
        phi[-1] *= 0.5
        s = np.asarray(s, dtype=np.float64)
        out = np.empty(s.shape, dtype=np.float64)
        flat_s, flat_out = s.ravel(), out.ravel()
        for start in range(0, flat_s.size, 256):   # chunked: (chunk x table) at a time
            chunk = flat_s[start:start + 256]
            flat_out[start:start + 256] = (np.cos(2.0 * np.pi * np.outer(chunk, d)) * phi).sum(axis=1) / self.resolution
        return out
