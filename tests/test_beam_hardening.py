"""preprocessing/beam_hardening.py: the polynomial correction, fitting it, and what it does to a reconstruction."""
import numpy as np
import pytest

from preprocessing import (
    coefficients_from_spectrum, correct_beam_hardening, fit_polynomial, is_monotonic, valid_attenuation_limit,
)
from preprocessing import beam_hardening as bh

# A polychromatic beam: bell-shaped 20-80 keV spectrum through an aluminium-like material (mu in 1/cm).
ENERGIES = np.linspace(20, 80, 61)
SPECTRUM = np.exp(-0.5 * ((ENERGIES - 45) / 12) ** 2)
MU = 0.2 + 12.0 * (ENERGIES / 20.0) ** -3


def transmission(attenuation):
    return np.exp(-np.asarray(attenuation, dtype=np.float64)).astype(np.float32)


def test_identity_coefficients_change_nothing():
    rng = np.random.default_rng(0)
    data = rng.uniform(0.05, 1.0, (4, 6, 8)).astype(np.float32)
    out = correct_beam_hardening(data, (1.0, 0.0, 0.0))
    assert out.dtype == np.float32 and out.shape == data.shape
    np.testing.assert_allclose(out, data, rtol=2e-6)


def test_known_mapping_and_input_untouched():
    p = np.array([0.0, 0.5, 1.0, 2.0])
    data = transmission(p)[None, None, :]
    before = data.copy()
    out = correct_beam_hardening(data, (1.0, 0.1))
    np.testing.assert_allclose(out[0, 0], np.exp(-(p + 0.1 * p ** 2)), rtol=2e-6)
    np.testing.assert_array_equal(data, before)


def test_values_at_or_below_zero_count_as_one_and_brighter_than_one_is_linear():
    data = np.array([[[0.0, -0.3, np.exp(0.5)]]], dtype=np.float32)
    out = correct_beam_hardening(data, (1.2, 0.3))[0, 0]
    assert out[0] == pytest.approx(1.0) and out[1] == pytest.approx(1.0)          # like the backends: <= 0 is 1
    assert out[2] == pytest.approx(np.exp(0.5 * 1.2), rel=1e-5)                    # p = -0.5 continues with slope c1


def test_beyond_the_valid_limit_the_map_continues_as_a_straight_line_and_keeps_increasing():
    c = (1.0, 0.0, -0.02)                      # flattens out near p = 4
    limit = valid_attenuation_limit(c)
    assert 3.0 < limit < 4.1
    p = np.linspace(0.0, 8.0, 200)
    g = bh.corrected_attenuation(p, c)
    assert np.all(np.diff(g) > 0)              # never turns around
    step = np.diff(g)
    assert np.allclose(step[p[:-1] > limit + 0.1], step[-1], rtol=1e-6)            # straight beyond the limit


def test_unusable_polynomials_are_refused_with_the_reason():
    data = transmission([[[0.5, 1.0]]])
    for bad in ((1.0, -0.5), (0.0, 1.0), (-1.0,), (1.0, 0.0, -0.2)):
        assert valid_attenuation_limit(bad) < bh.MIN_VALID_ATTENUATION
        with pytest.raises(ValueError, match="stop increasing"):
            correct_beam_hardening(data, bad)
    with pytest.raises(ValueError, match="at least one"):
        correct_beam_hardening(data, ())
    with pytest.raises(ValueError, match="finite"):
        correct_beam_hardening(data, (1.0, np.nan))
    assert is_monotonic((1.0, 0.3)) and not is_monotonic((1.0, -0.5))


def test_large_arrays_are_processed_in_blocks_with_the_same_result(monkeypatch):
    rng = np.random.default_rng(1)
    data = rng.uniform(0.05, 1.0, (5, 7, 9)).astype(np.float32)
    whole = correct_beam_hardening(data, (1.0, 0.2, -0.01))
    monkeypatch.setattr(bh, "_BLOCK_ELEMENTS", 17)
    np.testing.assert_array_equal(correct_beam_hardening(data, (1.0, 0.2, -0.01)), whole)


def test_fit_polynomial_recovers_exact_coefficients():
    p = np.linspace(0.0, 3.0, 40)
    target = 1.1 * p + 0.25 * p ** 2 - 0.03 * p ** 3
    np.testing.assert_allclose(fit_polynomial(p, target, order=3), (1.1, 0.25, -0.03), atol=1e-9)
    np.testing.assert_allclose(fit_polynomial(p, 2.0 * p, order=2), (2.0, 0.0), atol=1e-9)
    with pytest.raises(ValueError):
        fit_polynomial(p, target[:-1])
    with pytest.raises(ValueError):
        fit_polynomial(p[:2], target[:2], order=3)


def test_a_wide_spectrum_gives_concave_attenuation_and_a_monochromatic_one_does_not():
    thickness = np.linspace(0.0, 1.2, 50)
    poly = bh.simulate_polychromatic_attenuation(ENERGIES, SPECTRUM, MU, thickness)
    straight = poly[1] / thickness[1] * thickness
    assert np.all(poly[2:] < straight[2:])                      # hardened beam: less attenuation than a straight line
    mono = bh.simulate_polychromatic_attenuation(np.array([40.0]), np.array([1.0]), np.array([1.5]), thickness)
    np.testing.assert_allclose(mono, 1.5 * thickness, rtol=1e-12)
    np.testing.assert_allclose(coefficients_from_spectrum(np.array([40.0, 41.0]), np.array([1.0, 0.0]),
                                                          np.array([1.5, 1.4]), 1.2, order=2), (1.0, 0.0), atol=1e-6)


def test_coefficients_from_a_spectrum_linearize_the_attenuation():
    c = coefficients_from_spectrum(ENERGIES, SPECTRUM, MU, 1.2, order=3)
    assert c[0] > 0 and c[1] > 0                                 # thick parts get raised
    thickness = np.linspace(0.05, 1.2, 60)
    measured = bh.simulate_polychromatic_attenuation(ENERGIES, SPECTRUM, MU, thickness)
    corrected = bh.corrected_attenuation(measured, c)
    slope = corrected / thickness
    assert np.ptp(slope) / slope.mean() < 0.02                   # proportional to thickness again, within 2%
    raw_slope = measured / thickness
    assert np.ptp(raw_slope) / raw_slope.mean() > 0.2            # ... where the raw data are far from it
    assert valid_attenuation_limit(c) >= bh.MIN_VALID_ATTENUATION


def test_it_removes_cupping_from_a_reconstruction():
    """A uniform cylinder imaged with the polychromatic beam above: the middle reconstructs too dark;
    after the correction it is as bright as the edge."""
    pytest.importorskip("algotom")
    from reconstruction import Geometry, reconstruct

    width, radius_px, pixel_cm = 160, 60, 0.01
    s = np.arange(width) - (width - 1) / 2
    chord_cm = 2 * np.sqrt(np.clip(radius_px ** 2 - s ** 2, 0, None)) * pixel_cm
    t_poly = (SPECTRUM[None, :] / SPECTRUM.sum() * np.exp(-MU[None, :] * chord_cm[:, None])).sum(axis=1)
    angles = np.linspace(0, 180, 180, endpoint=False)

    def cupping_ratio(t_line):
        stack = np.repeat(t_line[None, None, :], 180, axis=0).astype(np.float32)
        image = reconstruct(stack[:, [0, 0], :], angles, (width - 1) / 2, geometry=Geometry(geometry_type="parallel"),
                            backend="algotom_fbp", device="cpu")[0]
        c = width // 2
        middle = image[c, c - 5:c + 5].mean()
        edge = image[c, c + int(0.75 * radius_px):c + int(0.9 * radius_px)].mean()
        return middle / edge

    uncorrected = cupping_ratio(t_poly)
    coefficients = coefficients_from_spectrum(ENERGIES, SPECTRUM, MU, chord_cm.max(), order=3)
    corrected = cupping_ratio(correct_beam_hardening(t_poly.reshape(1, 1, -1), coefficients).reshape(-1))
    assert uncorrected < 0.93                                    # cupping: the middle is >7% too dark
    assert corrected == pytest.approx(1.0, abs=0.03)
