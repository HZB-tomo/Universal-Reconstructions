"""preprocessing/phase.py: Paganin phase retrieval, its halo, and the method registry."""
import numpy as np
import pytest
import scipy.fft as sfft

from preprocessing import (
    PhaseMethod, PhaseParams, get_phase_method, list_phase_methods, paganin_filter, phase_halo_rows,
    register_phase_method, retrieve_phase,
)
from preprocessing import phase as phase_module

SETUP = dict(energy_kev=20.0, distance_mm=100.0, pixel_um=1.0, delta_over_beta=100.0)


def test_wavelength_from_energy():
    # 12.3984 keV is a wavelength of exactly 1 Angstrom = 1e-4 um
    assert PhaseParams(12.398419843320026, 1.0, 1.0, 1.0).wavelength_um == pytest.approx(1e-4)
    assert PhaseParams(20.0, 1.0, 1.0, 1.0).wavelength_um == pytest.approx(6.199e-5, rel=1e-3)


def test_invalid_parameters_are_refused():
    for bad in (dict(energy_kev=0), dict(energy_kev=-5), dict(distance_mm=-1), dict(pixel_um=0), dict(delta_over_beta=0)):
        with pytest.raises(ValueError):
            PhaseParams(**{**SETUP, **bad})
    PhaseParams(**{**SETUP, "distance_mm": 0.0})        # zero distance is allowed: nothing to undo


def test_filter_values_follow_the_paganin_formula():
    p = PhaseParams(**SETUP)
    response = paganin_filter((64, 64), p)
    assert response.shape == (64, 33) and response.dtype == np.float32
    assert response[0, 0] == 1.0                           # the mean level is untouched
    # u = 8 cycles over 64 pixels of 1 um -> 0.125 cycles/um
    u = 8 / 64
    expected = 1.0 / (1.0 + np.pi * p.wavelength_um * p.distance_um * 100.0 * u ** 2)
    assert response[0, 8] == pytest.approx(expected, rel=1e-6)
    assert response[8, 0] == pytest.approx(expected, rel=1e-6)           # same along the other axis
    assert np.all(response <= 1.0) and np.all(response > 0)


def test_stronger_conditions_smooth_more():
    base = paganin_filter((64, 64), PhaseParams(**SETUP))
    farther = paganin_filter((64, 64), PhaseParams(**{**SETUP, "distance_mm": 400.0}))
    softer = paganin_filter((64, 64), PhaseParams(**{**SETUP, "delta_over_beta": 400.0}))
    assert np.all(farther[:, 1:] < base[:, 1:]) and np.all(softer[:, 1:] < base[:, 1:])
    assert np.allclose(farther, softer)            # only the product distance * delta/beta matters


def test_a_flat_field_stays_flat_and_the_input_is_untouched():
    flat = np.full((6, 20, 24), 0.8, dtype=np.float32)
    before = flat.copy()
    out = retrieve_phase(flat, **SETUP)
    assert out.dtype == np.float32 and out.shape == flat.shape
    np.testing.assert_allclose(out, 0.8, rtol=1e-5)
    np.testing.assert_array_equal(flat, before)


def test_zero_distance_changes_nothing():
    rng = np.random.default_rng(0)
    data = rng.uniform(0.5, 1.0, (3, 16, 16)).astype(np.float32)
    np.testing.assert_array_equal(retrieve_phase(data, **{**SETUP, "distance_mm": 0.0}), data)


def test_it_removes_edge_fringes_from_a_simulated_propagation_image():
    """A sphere of one material, imaged after 100 mm of free-space propagation: the raw
    -log(I) is full of fringes; Paganin gives back the projected attenuation."""
    p = PhaseParams(**SETUP)
    lam, z = p.wavelength_um, p.distance_um
    beta = 1e-8
    delta = p.delta_over_beta * beta
    k = 2 * np.pi / lam
    n = 256
    y, x = np.mgrid[:n, :n] - n / 2
    r2 = x ** 2 + y ** 2
    radius = 50.0
    thickness = np.where(r2 < radius ** 2, 2 * np.sqrt(np.clip(radius ** 2 - r2, 0, None)), 0.0)    # um
    wave = np.exp(-beta * k * thickness - 1j * delta * k * thickness)
    u = np.fft.fftfreq(n, d=1.0)
    propagator = np.exp(-1j * np.pi * lam * z * (u[:, None] ** 2 + u[None, :] ** 2))
    intensity = np.abs(np.fft.ifft2(np.fft.fft2(wave) * propagator)) ** 2

    truth = 2 * k * beta * thickness                       # mu * T
    raw = -np.log(np.clip(intensity, 1e-6, None))
    retrieved = -np.log(np.clip(retrieve_phase(intensity[None].astype(np.float32), **SETUP)[0], 1e-6, None))
    error = lambda a: np.sqrt(np.mean((a - truth) ** 2)) / truth.max()
    assert error(raw) > 0.2                                # the fringes are a large error
    assert error(retrieved) < 0.03                         # ... and Paganin removes nearly all of it
    inside = r2 < (0.8 * radius) ** 2
    assert retrieved[inside].mean() / truth[inside].mean() == pytest.approx(1.0, abs=0.03)


def test_halo_is_enough_for_a_block_to_match_the_whole_image():
    rng = np.random.default_rng(1)
    base = 0.7 + 0.2 * rng.random((4, 110, 40))
    stack = base.astype(np.float32)
    setup = dict(energy_kev=25.0, distance_mm=100.0, pixel_um=2.0, delta_over_beta=150.0)
    halo = phase_halo_rows(**setup)
    assert 2 <= halo <= 40
    whole = retrieve_phase(stack, **setup)
    lo, hi = 40, 64
    block = retrieve_phase(stack[:, lo - halo:hi + halo], **setup)[:, halo:-halo]
    np.testing.assert_allclose(block, whole[:, lo:hi], atol=2e-3 * np.ptp(whole))
    # without a halo the block edges are visibly wrong -- that is what the halo is for
    bare = retrieve_phase(stack[:, lo:hi], **setup)
    assert np.abs(bare - whole[:, lo:hi]).max() > 5 * np.abs(block - whole[:, lo:hi]).max()


def test_halo_grows_with_the_smoothing_length():
    small = phase_halo_rows(**SETUP)
    large = phase_halo_rows(**{**SETUP, "distance_mm": 1600.0})
    assert large > small >= 1
    assert phase_halo_rows(**{**SETUP, "distance_mm": 0.0}) == 0


def test_batches_of_angles_do_not_change_the_result():
    rng = np.random.default_rng(2)
    data = rng.uniform(0.4, 1.0, (9, 24, 28)).astype(np.float32)
    whole = retrieve_phase(data, **SETUP)
    batched = retrieve_phase(data, **SETUP, max_batch_bytes=1)       # one projection at a time
    np.testing.assert_allclose(batched, whole, rtol=1e-6, atol=1e-7)


def test_the_registry_lists_paganin_and_takes_new_methods():
    assert "paganin" in list_phase_methods()
    assert get_phase_method("paganin").name == "paganin"

    @register_phase_method
    class Doubler(PhaseMethod):
        name = "doubler_for_tests"

        def halo_rows(self, params, **options):
            return 3

        def retrieve(self, projections, params, **options):
            return np.asarray(projections, dtype=np.float32) * 2

    try:
        assert "doubler_for_tests" in list_phase_methods()
        out = retrieve_phase(np.ones((1, 4, 4), dtype=np.float32), method="doubler_for_tests", **SETUP)
        assert float(out[0, 0, 0]) == 2.0
        assert phase_halo_rows(method="doubler_for_tests", **SETUP) == 3
    finally:
        phase_module._REGISTRY.pop("doubler_for_tests", None)


def test_unknown_method_and_bad_shape():
    with pytest.raises(KeyError, match="Registered"):
        retrieve_phase(np.ones((1, 4, 4), dtype=np.float32), method="nope", **SETUP)
    with pytest.raises(ValueError, match="angles, rows, width"):
        retrieve_phase(np.ones((4, 4), dtype=np.float32), **SETUP)
