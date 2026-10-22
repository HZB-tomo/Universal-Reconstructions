"""Reconstruction filters: the ramp |k| (with the Ram-Lak DC term) times an optional smoothing window."""
from __future__ import annotations

import numpy as np

FILTERS = ("ramp", "hann", "hamming", "cosine", "shepp-logan")


def filter_response(name: str, n: int) -> np.ndarray:
    """Filter for the non-negative frequency bins k = 0 .. n/2 of a length-n
    FFT, as |k| times a window that falls to its end value at Nyquist
    (k = n/2). Same meaning as FBP's `filter_name`; "ramp" is unwindowed."""
    if name not in FILTERS:
        raise ValueError(f"unknown filter {name!r}; choose from {FILTERS}")
    k = np.arange(n // 2 + 1, dtype=np.float64)
    kappa = k / (n / 2.0)   # 0 .. 1
    if name == "ramp":
        window = np.ones_like(kappa)
    elif name == "hann":
        window = 0.5 * (1.0 + np.cos(np.pi * kappa))
    elif name == "hamming":
        window = 0.54 + 0.46 * np.cos(np.pi * kappa)
    elif name == "cosine":
        window = np.cos(0.5 * np.pi * kappa)
    else:  # shepp-logan
        window = np.sinc(0.5 * kappa)
    response = k * window
    # The ideal ramp is zero at k = 0, which loses the image's mean level (about
    # -2.5% inside objects). The sampled Ram-Lak kernel of filtered backprojection
    # (Kak & Slaney) has a small positive DC response, N * H(0) = 2 / pi^2 in these
    # units, independent of N. Using it makes the amplitudes agree with FBP.
    response[0] = 2.0 / np.pi ** 2
    return response
