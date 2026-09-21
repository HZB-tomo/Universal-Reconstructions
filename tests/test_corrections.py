"""
Tests for preprocessing/corrections.py: real correctness against the
actual algotom library where it's available (skipped entirely if it
isn't -- this project doesn't require algotom to be installed), plus
the friendly-ImportError fallback when it's genuinely missing, tested
by making the exact module algotom imports unavailable via sys.modules
rather than actually uninstalling anything.
"""
import sys

import numpy as np
import pytest

from preprocessing import corrections

algotom = pytest.importorskip("algotom", reason="algotom not installed -- corrections.py is optional")


def _sinogram(seed=0, n_angles=40, width=48):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 1.0, size=(n_angles, width)).astype(np.float32)


def _volume(seed=0, n_angles=20, height=6, width=48):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 1.0, size=(n_angles, height, width)).astype(np.float32)


def test_remove_zinger_preserves_shape():
    sino = _sinogram()

    result = corrections.remove_zinger(sino)

    assert result.shape == sino.shape


def test_remove_zinger_actually_removes_a_spike():
    sino = _sinogram()
    spiked = sino.copy()
    spiked[10, 20] += 50.0  # a single-pixel outlier, well above the rest

    result = corrections.remove_zinger(spiked, threshold=0.1)

    assert result[10, 20] < spiked[10, 20]


@pytest.mark.parametrize("method", [
    "sorting", "filtering", "fitting", "normalization", "all",
    "large_stripe", "dead_stripe", "interpolation", "2d_filtering_sorting",
    "fft", "wavelet_fft", "regularization",
])
def test_remove_ring_artifacts_preserves_shape_for_every_method(method):
    sino = _sinogram()

    result = corrections.remove_ring_artifacts(sino, method=method)

    assert result.shape == sino.shape


def test_process_volume_matches_per_slice_calls_for_zinger():
    volume = _volume()

    batched = corrections.process_volume(volume, corrections.remove_zinger, 0.08)

    for row in range(volume.shape[1]):
        one_slice = corrections.remove_zinger(volume[:, row, :], 0.08)
        np.testing.assert_allclose(batched[:, row, :], one_slice, rtol=1e-4)


def test_process_volume_matches_per_slice_calls_for_ring_removal():
    volume = _volume()

    batched = corrections.process_volume(volume, corrections.remove_ring_artifacts, method="sorting")

    for row in range(volume.shape[1]):
        one_slice = corrections.remove_ring_artifacts(volume[:, row, :], method="sorting")
        np.testing.assert_allclose(batched[:, row, :], one_slice, rtol=1e-4)


def test_remove_zinger_friendly_error_without_algotom(monkeypatch):
    monkeypatch.setitem(sys.modules, "algotom.prep.removal", None)

    with pytest.raises(ImportError, match="pip install algotom"):
        corrections.remove_zinger(np.zeros((4, 4), dtype=np.float32))


def test_remove_ring_artifacts_friendly_error_without_algotom(monkeypatch):
    monkeypatch.setitem(sys.modules, "algotom.prep.removal", None)

    with pytest.raises(ImportError, match="pip install algotom"):
        corrections.remove_ring_artifacts(np.zeros((4, 4), dtype=np.float32))


def test_process_volume_friendly_error_without_algotom(monkeypatch):
    monkeypatch.setitem(sys.modules, "algotom.util.utility", None)

    with pytest.raises(ImportError, match="pip install algotom"):
        corrections.process_volume(np.zeros((3, 4, 4), dtype=np.float32), corrections.remove_zinger, 0.08)
