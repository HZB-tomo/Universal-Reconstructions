"""
Beam hardening correction by polynomial linearization.

Physics. Filtered backprojection assumes Beer-Lambert's law for one X-ray
energy: the measured attenuation p = -ln(I / I0) grows in proportion to the
material thickness L (p = mu * L). A real beam, from a laboratory tube or a
white/pink synchrotron beam, is a spread of energies. Low-energy photons are
absorbed first, so the beam that has passed through more material has a higher
average energy ("hardened") and is absorbed less per cm:

    p(L) = -ln( integral S(E) exp(-mu(E) L) dE / integral S(E) dE )

is concave in L, not a straight line. Reconstructed as if it were linear, the
middle of a uniform object looks too dark ("cupping"), and dense parts cast
dark streaks onto their neighbours. (Neutron beams from a reactor show the
same effect.)

The correction replaces the measured attenuation by a polynomial of it,

    p_corrected = c1*p + c2*p^2 + c3*p^3 + ...

chosen so that p_corrected is proportional to thickness again. For one dominant
material this removes cupping almost completely. Where the coefficients come
from:

  * a calibration: measured attenuation of known thicknesses (a step wedge, a
    cylinder) against the ideal straight line -> fit_polynomial();
  * a model of the beam: the spectrum and the material's mu(E) ->
    coefficients_from_spectrum();
  * by eye: raise c2 until the centre of a uniform object in a preview is as
    bright as its edge (the GUI's controls).

Limits. It linearizes for one material; with two very different materials (bone
and soft tissue) one polynomial cannot be right for both, and an iterative
correction is needed. The polynomial must be increasing over the attenuation
range of the data, or two different thicknesses would map to the same value. A
fitted cubic usually turns over somewhere beyond the range it was fitted on;
that is fine, the correction simply continues as a straight line from the last
point where it is still increasing (valid_attenuation_limit()), as long as that
point is at least MIN_VALID_ATTENUATION.

Pipeline contract. Like the other preprocessing steps this takes and returns
TRANSMISSION (the reconstruction backends take the -log themselves): the
correction is exp(-g(-ln T)) applied to every value. Values <= 0 count as 1
(like the backends); T > 1 (negative attenuation, noise) is continued linearly
with slope c1; attenuation above the polynomial's valid limit is continued linearly
with its slope there, so the map never turns around.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Attenuation (-ln T) up to which the polynomial is applied and checked. 8 is
# T = 3e-4: beyond anything a usable projection contains.
DEFAULT_MAX_ATTENUATION = 8.0
# The polynomial must be increasing at least up to here (T = 5%), or it is refused.
MIN_VALID_ATTENUATION = 3.0
_BLOCK_ELEMENTS = 4_000_000   # elements processed at a time (bounds the temporaries)


def _coefficients(coefficients: Sequence[float]) -> np.ndarray:
    c = np.asarray(coefficients, dtype=np.float64).ravel()
    if c.size == 0:
        raise ValueError("beam hardening needs at least one coefficient (c1)")
    if not np.all(np.isfinite(c)):
        raise ValueError(f"beam hardening coefficients must be finite numbers, got {c.tolist()}")
    return c


def polynomial(p: np.ndarray, coefficients: Sequence[float]) -> np.ndarray:
    """c1*p + c2*p^2 + ... (Horner's rule; no constant term: zero attenuation stays zero)."""
    c = _coefficients(coefficients)
    out = np.zeros_like(np.asarray(p, dtype=np.float64))
    for ck in c[::-1]:
        out = (out + ck) * p
    return out


def derivative(p: np.ndarray, coefficients: Sequence[float]) -> np.ndarray:
    c = _coefficients(coefficients)
    k = np.arange(1, c.size + 1)
    return polynomial_from_powers(p, k * c, start_power=0)


def polynomial_from_powers(p: np.ndarray, weights: np.ndarray, start_power: int) -> np.ndarray:
    out = np.zeros_like(np.asarray(p, dtype=np.float64))
    for w in weights[::-1]:
        out = out * p + w
    return out * np.asarray(p, dtype=np.float64) ** start_power


def is_monotonic(coefficients: Sequence[float], max_attenuation: float = DEFAULT_MAX_ATTENUATION) -> bool:
    """True if the correction is strictly increasing on [0, max_attenuation]."""
    grid = np.linspace(0.0, max_attenuation, 513)
    return bool(np.all(derivative(grid, coefficients) > 0.0))


def valid_attenuation_limit(coefficients: Sequence[float], max_attenuation: float = DEFAULT_MAX_ATTENUATION) -> float:
    """The largest attenuation up to which the polynomial is increasing at least a twentieth
    as fast as it starts (at most max_attenuation; the correction continues as a straight line
    from there, so it must not have flattened out): 0 if it is not even increasing at the start."""
    c = _coefficients(coefficients)
    grid = np.linspace(0.0, max_attenuation, 2049)
    positive = (derivative(grid, c) > 0.05 * c[0]) & (c[0] > 0.0)
    if positive.all():
        return float(max_attenuation)
    first_bad = int(np.argmin(positive))        # first index where the derivative is not positive
    return float(grid[first_bad - 1]) if first_bad > 0 else 0.0


def corrected_attenuation(
    attenuation: np.ndarray, coefficients: Sequence[float], max_attenuation: float = DEFAULT_MAX_ATTENUATION,
) -> np.ndarray:
    """The correction g(p) on attenuation values, with the linear continuations described in the module
    docstring: below 0 with slope c1, above the valid limit with the polynomial's slope there."""
    c = _coefficients(coefficients)
    p = np.asarray(attenuation, dtype=np.float64)
    limit = valid_attenuation_limit(c, max_attenuation)
    clipped = np.clip(p, 0.0, limit)
    slope_high = float(derivative(np.array(limit), c))
    slope = np.where(p < 0.0, c[0], np.where(p > limit, slope_high, 0.0))
    return polynomial(clipped, c) + slope * (p - clipped)


def correct_beam_hardening(
    projections: np.ndarray, coefficients: Sequence[float], *, max_attenuation: float = DEFAULT_MAX_ATTENUATION,
) -> np.ndarray:
    """
    Linearize beam-hardened transmission projections (any shape; flat/dark-normalized values around
    0..1). Returns the corrected transmission, float32, same shape; the input is not modified.

        correct_beam_hardening(T, coefficients=(1.0, 0.35, -0.05))

    Raises ValueError if the polynomial is not increasing up to MIN_VALID_ATTENUATION (it may turn over
    further out; it is then continued as a straight line).
    """
    c = _coefficients(coefficients)
    limit = valid_attenuation_limit(c, max_attenuation)
    if limit < min(MIN_VALID_ATTENUATION, max_attenuation):
        raise ValueError(
            f"beam hardening coefficients {c.tolist()} stop increasing at attenuation {limit:.2f} "
            f"(need at least {MIN_VALID_ATTENUATION:g}): two different thicknesses would map to the same "
            "value. Reduce the higher-order terms."
        )
    data = np.asarray(projections)
    out = np.empty(data.shape, dtype=np.float32)
    flat_in, flat_out = data.reshape(-1), out.reshape(-1)
    for start in range(0, flat_in.size, _BLOCK_ELEMENTS):
        block = flat_in[start:start + _BLOCK_ELEMENTS].astype(np.float64)
        block[block <= 0.0] = 1.0                              # like the reconstruction backends
        p = corrected_attenuation(-np.log(block), c, max_attenuation)
        flat_out[start:start + _BLOCK_ELEMENTS] = np.exp(-p)
    return out


# -- finding the coefficients ---------------------------------------------------------------------

def fit_polynomial(measured: np.ndarray, target: np.ndarray, order: int = 3) -> tuple:
    """
    Least-squares coefficients (c1 .. c_order) with  sum c_k * measured^k ~ target.

    `measured`: attenuation measured through known thicknesses (a step wedge, a cylinder's
    chord lengths); `target`: what it should be if the beam were monochromatic, i.e. a straight
    line through the origin (mu_reference * thickness). Same shape.
    """
    x = np.asarray(measured, dtype=np.float64).ravel()
    y = np.asarray(target, dtype=np.float64).ravel()
    if x.shape != y.shape or x.size < order:
        raise ValueError(f"need matching measured/target values, at least {order} of them (got {x.size} and {y.size})")
    design = np.stack([x ** k for k in range(1, order + 1)], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return tuple(float(v) for v in coefficients)


def simulate_polychromatic_attenuation(
    energies_kev: np.ndarray, spectrum: np.ndarray, mu_per_cm: np.ndarray, thickness_cm: np.ndarray,
) -> np.ndarray:
    """p(L) = -ln( sum S(E) exp(-mu(E) L) / sum S(E) ) for a beam of spectrum S through thicknesses L."""
    weights = np.asarray(spectrum, dtype=np.float64)
    weights = weights / weights.sum()
    mu = np.asarray(mu_per_cm, dtype=np.float64)
    thickness = np.asarray(thickness_cm, dtype=np.float64)
    transmission = (weights[None, :] * np.exp(-mu[None, :] * thickness.reshape(-1, 1))).sum(axis=1)
    return (-np.log(transmission)).reshape(thickness.shape)


def coefficients_from_spectrum(
    energies_kev: np.ndarray, spectrum: np.ndarray, mu_per_cm: np.ndarray, max_thickness_cm: float,
    *, order: int = 3, reference_energy_kev: Optional[float] = None, n_points: int = 256,
) -> tuple:
    """
    Polynomial coefficients for a beam with the given spectrum through one material.

    `energies_kev`, `spectrum` (relative photon weights, detector response included) and
    `mu_per_cm` (the material's linear attenuation at each energy, e.g. from NIST tables) describe
    the beam; `max_thickness_cm` is the largest thickness the data contain. The target is a
    straight line with slope mu(reference_energy), or, by default, the slope of the measured
    attenuation at vanishing thickness (the beam's initial effective attenuation).
    """
    energies = np.asarray(energies_kev, dtype=np.float64)
    mu = np.asarray(mu_per_cm, dtype=np.float64)
    if not (energies.shape == np.shape(spectrum) == mu.shape):
        raise ValueError("energies, spectrum and mu_per_cm must have the same shape")
    thickness = np.linspace(0.0, float(max_thickness_cm), n_points)
    measured = simulate_polychromatic_attenuation(energies, spectrum, mu, thickness)
    if reference_energy_kev is None:
        step = max_thickness_cm / (n_points * 100.0)
        mu_ref = float(simulate_polychromatic_attenuation(energies, spectrum, mu, np.array([step]))[0] / step)
    else:
        mu_ref = float(np.interp(reference_energy_kev, energies, mu))
    return fit_polynomial(measured, mu_ref * thickness, order=order)
