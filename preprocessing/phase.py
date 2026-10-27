"""
Phase retrieval for propagation-based phase-contrast tomography.

Physics. With a coherent beam (synchrotron), a sample changes the phase of the
wave as well as its amplitude. After the wave travels a distance z to the
detector those phase changes become intensity fringes: bright and dark edges
along every boundary ("edge enhancement"). Left alone they leave halos and
wrong values in the reconstruction. Phase retrieval turns the fringed image
back into a projected thickness that obeys Beer-Lambert's law, so ordinary
filtered backprojection can reconstruct it.

Paganin's method (Paganin et al., J. Microsc. 206, 33, 2002) assumes the
sample is one homogeneous material, with refractive index 1 - delta + i beta.
Then, with I the flat/dark-normalized intensity (I0 = 1),

    exp(-mu T) = F^-1 [ F[I] / (1 + pi * lambda * z * (delta/beta) * u^2) ]

where T is the projected thickness, mu = 4 pi beta / lambda the attenuation
coefficient, lambda the wavelength, z the sample-to-detector distance and u
the spatial frequency (cycles per length). That is a low-pass filter applied
to every projection, with one material-dependent number delta/beta. The
result is "transmission" again, so the usual -log() inside the reconstruction
backends gives mu * T. The price is resolution: the larger z * delta/beta,
the more is smoothed (the noise drops by the same token).

Because the filter mixes neighbouring detector rows, phase retrieval is a 2D
operation. Code that works on a few rows at a time (chunked/streamed
reconstruction, single-slice previews) must read a few extra rows on each
side, the "halo"; see PhaseMethod.halo_rows().

Assumptions: monochromatic beam, single material (or a delta/beta that is a
good average), weak absorption and slowly varying phase, flat/dark-corrected
input. It is only as good as the delta/beta you give it; it is a tuning knob
in practice.

Methods register like reconstruction backends (register_phase_method), so
others (Bronnikov, multi-distance CTF, ...) can be added without touching the
pipeline.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import scipy.fft as sfft

# h * c in keV * Angstrom: wavelength [Angstrom] = HC / energy [keV].
HC_KEV_ANGSTROM = 12.398419843320026
_ANGSTROM_IN_UM = 1.0e-4
_MM_IN_UM = 1.0e3


@dataclass(frozen=True)
class PhaseParams:
    """The physical setup phase retrieval needs. Lengths in the units named."""

    energy_kev: float
    distance_mm: float          # sample-to-detector propagation distance
    pixel_um: float            # effective pixel size at the sample
    delta_over_beta: float      # ratio of refractive-index decrement to absorption index

    def __post_init__(self) -> None:
        if not self.energy_kev > 0:
            raise ValueError(f"energy must be positive, got {self.energy_kev!r} keV")
        if not self.distance_mm >= 0:
            raise ValueError(f"propagation distance must be zero or positive, got {self.distance_mm!r} mm")
        if not self.pixel_um > 0:
            raise ValueError(f"pixel size must be positive, got {self.pixel_um!r} um")
        if not self.delta_over_beta > 0:
            raise ValueError(f"delta/beta must be positive, got {self.delta_over_beta!r}")

    @property
    def wavelength_um(self) -> float:
        return HC_KEV_ANGSTROM / self.energy_kev * _ANGSTROM_IN_UM

    @property
    def distance_um(self) -> float:
        return self.distance_mm * _MM_IN_UM

    @property
    def strength_um2(self) -> float:
        """lambda * z * delta/beta: the filter is 1 / (1 + pi * strength * u^2)."""
        return self.wavelength_um * self.distance_um * self.delta_over_beta

    @property
    def blur_length_um(self) -> float:
        """sqrt(lambda z (delta/beta) / (4 pi)): the length scale of the smoothing
        (the decay length of the filter's real-space kernel)."""
        return math.sqrt(self.strength_um2 / (4.0 * math.pi))


class PhaseMethod(ABC):
    """
    Base class for a phase retrieval method. `name` is the registry key.

    retrieve() takes normalized transmission projections, shape
    (n_angles, n_rows, width), and returns an array of the same shape that is
    again a transmission (so -log gives the retrieved attenuation).
    halo_rows() is how many extra rows beyond a block's own edges the method
    needs to give the same answer for the block as for the whole image.
    """

    name: str

    @abstractmethod
    def halo_rows(self, params: PhaseParams, **options) -> int:
        """Rows needed above and below a block of rows."""

    @abstractmethod
    def retrieve(self, projections: np.ndarray, params: PhaseParams, **options) -> np.ndarray:
        """Phase-retrieved transmission, same shape, float32. Must not modify the input."""


_REGISTRY: dict[str, PhaseMethod] = {}


def register_phase_method(cls):
    """Decorator: instantiates a PhaseMethod subclass and adds it to the registry."""
    instance = cls()
    _REGISTRY[instance.name] = instance
    return cls


def get_phase_method(name: str) -> PhaseMethod:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown phase retrieval method '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_phase_methods() -> list[str]:
    return sorted(_REGISTRY)


def paganin_filter(shape: tuple[int, int], params: PhaseParams) -> np.ndarray:
    """
    The Paganin filter 1 / (1 + pi lambda z (delta/beta) u^2) on the frequency grid of a
    real 2D FFT of a (rows, cols) image: shape (rows, cols // 2 + 1), float32.
    """
    rows, cols = shape
    fy = sfft.fftfreq(rows, d=params.pixel_um)         # cycles per um
    fx = sfft.rfftfreq(cols, d=params.pixel_um)
    u2 = fy[:, None] ** 2 + fx[None, :] ** 2
    return (1.0 / (1.0 + math.pi * params.strength_um2 * u2)).astype(np.float32)


@register_phase_method
class PaganinMethod(PhaseMethod):
    name = "paganin"

    #: Real-space kernel of the filter decays like exp(-r / blur_length); this many
    #: blur lengths of halo leave about e^-8 = 3e-4 of its weight outside.
    HALO_BLUR_LENGTHS = 8.0
    #: Upper bound on the working memory of one batch of padded projections.
    MAX_BATCH_BYTES = 256 * 1024 ** 2

    def halo_rows(self, params: PhaseParams, halo_blur_lengths: float = HALO_BLUR_LENGTHS, **_ignored) -> int:
        return int(math.ceil(halo_blur_lengths * params.blur_length_um / params.pixel_um))

    def retrieve(
        self, projections: np.ndarray, params: PhaseParams, pad: int | None = None,
        max_batch_bytes: int | None = None, halo_blur_lengths: float = HALO_BLUR_LENGTHS, **_ignored,
    ) -> np.ndarray:
        """
        `pad`: edge-replicated pixels added on every side before the FFT so the filter does
        not wrap around (default: the halo). Projections are processed in batches of angles
        small enough for `max_batch_bytes`.
        """
        data = np.asarray(projections)
        if data.ndim != 3:
            raise ValueError(f"expected projections of shape (angles, rows, width), got {data.shape}")
        if params.strength_um2 == 0.0:
            return np.array(data, dtype=np.float32)          # no propagation, nothing to undo
        n_angles, n_rows, width = data.shape
        if pad is None:
            pad = self.halo_rows(params, halo_blur_lengths=halo_blur_lengths)
        rows_padded = sfft.next_fast_len(n_rows + 2 * pad, real=True)
        cols_padded = sfft.next_fast_len(width + 2 * pad, real=True)
        response = paganin_filter((rows_padded, cols_padded), params)
        before_r, before_c = pad, pad
        after_r, after_c = rows_padded - n_rows - before_r, cols_padded - width - before_c

        per_projection = rows_padded * cols_padded * 8      # float32 in + complex64 spectrum, roughly
        batch = max(1, int((max_batch_bytes or self.MAX_BATCH_BYTES) // per_projection))
        out = np.empty((n_angles, n_rows, width), dtype=np.float32)
        for start in range(0, n_angles, batch):
            block = np.pad(
                data[start:start + batch].astype(np.float32, copy=False),
                ((0, 0), (before_r, after_r), (before_c, after_c)), mode="edge",
            )
            spectrum = sfft.rfft2(block, axes=(1, 2), workers=-1)
            spectrum *= response
            filtered = sfft.irfft2(spectrum, s=(rows_padded, cols_padded), axes=(1, 2), workers=-1)
            out[start:start + batch] = filtered[:, before_r:before_r + n_rows, before_c:before_c + width]
        return out


def retrieve_phase(
    projections: np.ndarray, *, energy_kev: float, distance_mm: float, pixel_um: float,
    delta_over_beta: float, method: str = "paganin", **options,
) -> np.ndarray:
    """
    Phase-retrieve flat/dark-normalized transmission projections (angles, rows, width).

        retrieve_phase(T, energy_kev=20, distance_mm=100, pixel_um=1.1, delta_over_beta=200)

    Returns the retrieved transmission, float32, same shape: pass it on to the
    reconstruction, whose -log gives the attenuation. When working on a block of rows of a
    bigger image, include `phase_halo_rows(...)` extra rows on each side and drop them again
    afterwards.
    """
    params = PhaseParams(energy_kev, distance_mm, pixel_um, delta_over_beta)
    return get_phase_method(method).retrieve(projections, params, **options)


def phase_halo_rows(
    *, energy_kev: float, distance_mm: float, pixel_um: float, delta_over_beta: float,
    method: str = "paganin", **options,
) -> int:
    """Extra rows needed on each side of a block for retrieve_phase() to match the whole image."""
    params = PhaseParams(energy_kev, distance_mm, pixel_um, delta_over_beta)
    return get_phase_method(method).halo_rows(params, **options)
