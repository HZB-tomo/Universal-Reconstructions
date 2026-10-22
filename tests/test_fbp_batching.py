"""
algotom_fbp reconstructs a whole stack in one (GPU-friendly) call when all
slices share a center, and falls back to per-slice calls with a shared
ramp filter when centers differ. Both must match plain per-slice FBP.
The GPU kernel itself can't run on CI; its batch planning is tested directly.
"""
import numpy as np
import pytest

from orchestration import ram_check
from reconstruction.backend import Geometry, get_backend
from reconstruction.backends.algotom_fbp import gpu_batch_size, plan_batches

algotom = pytest.importorskip("algotom")
from algotom.rec import reconstruction as rec  # noqa: E402

GEOMETRY = Geometry(geometry_type="parallel")


@pytest.fixture
def stack():
    rng = np.random.default_rng(0)
    n_angles, n_slices, width = 90, 5, 64
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    return rng.uniform(0.2, 1.0, (n_angles, n_slices, width)).astype(np.float32), angles


def _assert_close(actual, expected):
    # Border pixels sit exactly on the detector edge at some angles, where
    # float32 vs float64 rounding of the ray position includes or drops one
    # sinogram sample; compare the interior.
    actual, expected = actual[:, 1:-1, 1:-1], expected[:, 1:-1, 1:-1]
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 3e-3


def _serial_reference(stack, angles, centers):
    """algotom's own FBP with its backprojection run as plain Python (no
    prange, so no data race): the correct answer, slowly."""
    from unittest import mock

    with mock.patch.object(rec, "back_projection_cpu", rec.back_projection_cpu.py_func):
        return np.stack([
            rec.fbp_reconstruction(stack[:, i, :].copy(), float(c), angles=np.radians(angles), gpu=False)
            for i, c in enumerate(centers)
        ])


def test_cpu_fbp_matches_a_serial_reference(stack):
    sinograms, angles = stack
    out = get_backend("algotom_fbp").reconstruct(sinograms.copy(), angles, 31.5, GEOMETRY, device="cpu")
    assert out.shape == (5, 64, 64)
    _assert_close(out, _serial_reference(sinograms, angles, [31.5] * 5))


def test_cpu_fbp_with_different_centers_matches_a_serial_reference(stack):
    sinograms, angles = stack
    centers = np.array([30.0, 30.5, 31.0, 31.5, 32.0])
    out = get_backend("algotom_fbp").reconstruct(sinograms.copy(), angles, centers, GEOMETRY, device="cpu")
    _assert_close(out, _serial_reference(sinograms, angles, centers))


def test_cpu_fbp_is_exactly_reproducible_and_leaves_its_input_alone(stack):
    sinograms, angles = stack
    backend = get_backend("algotom_fbp")
    before = sinograms.copy()
    first = backend.reconstruct(sinograms, angles, 31.5, GEOMETRY, device="cpu")
    second = backend.reconstruct(sinograms, angles, 31.5, GEOMETRY, device="cpu")
    np.testing.assert_array_equal(first, second)  # algotom's own CPU backprojection fails this
    np.testing.assert_array_equal(sinograms, before)


def test_single_slice_keeps_the_stack_shape(stack):
    sinograms, angles = stack
    out = get_backend("algotom_fbp").reconstruct(sinograms[:, :1].copy(), angles, 31.5, GEOMETRY, device="cpu")
    assert out.shape == (1, 64, 64)


def test_plan_batches_covers_every_slice_once():
    assert plan_batches(10, 4) == [slice(0, 4), slice(4, 8), slice(8, 10)]
    assert plan_batches(3, 100) == [slice(0, 3)]
    assert plan_batches(3, 0) == [slice(0, 1), slice(1, 2), slice(2, 3)]  # never a zero-size batch


def test_gpu_batch_size_scales_with_free_memory():
    assert gpu_batch_size(1000, 1024, None) == 16
    small = gpu_batch_size(1000, 1024, 1 * 1024**3)
    large = gpu_batch_size(1000, 1024, 16 * 1024**3)
    assert 1 <= small < large
    assert gpu_batch_size(100000, 100000, 1) == 1  # never below one slice


class _Mem:
    available = 64 * 1024**3


def test_gpu_chunks_are_bigger_than_cpu_chunks_when_gpu_memory_allows(monkeypatch):
    monkeypatch.setattr(ram_check.psutil, "virtual_memory", lambda: _Mem())
    monkeypatch.setattr(ram_check.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(ram_check, "gpu_free_bytes", lambda: 24 * 1024**3)
    cpu = ram_check.estimate_chunk_rows(1024, 5000, "cpu", 1000)
    gpu = ram_check.estimate_chunk_rows(1024, 5000, "gpu", 1000)
    assert cpu == 8 and 8 < gpu <= ram_check.MAX_GPU_CHUNK_ROWS


def test_gpu_chunks_fall_back_to_cpu_sizing_without_gpu_info(monkeypatch):
    monkeypatch.setattr(ram_check.psutil, "virtual_memory", lambda: _Mem())
    monkeypatch.setattr(ram_check.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(ram_check, "gpu_free_bytes", lambda: None)
    assert ram_check.estimate_chunk_rows(1024, 5000, "gpu", 1000) == 8
    assert ram_check.estimate_chunk_rows(1024, 3, "gpu", 1000) == 3


# -- BPF and DFI ---------------------------------------------------------------


def _bpf_serial_reference(stack, angles, centers):
    from unittest import mock

    with mock.patch.object(rec, "back_projection_cpu", rec.back_projection_cpu.py_func):
        return np.stack([
            rec.bpf_reconstruction(stack[:, i, :].copy(), float(c), angles=np.radians(angles), gpu=False)
            for i, c in enumerate(centers)
        ])


def test_cpu_bpf_matches_a_serial_reference(stack):
    sinograms, angles = stack
    out = get_backend("algotom_bpf").reconstruct(sinograms.copy(), angles, 31.5, GEOMETRY, device="cpu")
    assert out.shape == (5, 64, 64)
    _assert_close(out, _bpf_serial_reference(sinograms, angles, [31.5] * 5))


def test_cpu_bpf_with_different_centers_matches_a_serial_reference(stack):
    sinograms, angles = stack
    centers = np.array([30.0, 30.5, 31.0, 31.5, 32.0])
    out = get_backend("algotom_bpf").reconstruct(sinograms.copy(), angles, centers, GEOMETRY, device="cpu")
    _assert_close(out, _bpf_serial_reference(sinograms, angles, centers))


def test_cpu_bpf_is_exactly_reproducible_and_leaves_its_input_alone(stack):
    sinograms, angles = stack
    backend = get_backend("algotom_bpf")
    before = sinograms.copy()
    first = backend.reconstruct(sinograms, angles, 31.5, GEOMETRY, device="cpu")
    second = backend.reconstruct(sinograms, angles, 31.5, GEOMETRY, device="cpu")
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(sinograms, before)


def _dfi_per_slice(stack, angles, centers):
    return np.stack([
        rec.dfi_reconstruction(stack[:, i, :].copy(), float(c), angles=np.radians(angles))
        for i, c in enumerate(centers)
    ])


def test_dfi_batched_matches_per_slice(stack):
    sinograms, angles = stack
    out = get_backend("algotom_dfi").reconstruct(sinograms.copy(), angles, 31.5, GEOMETRY)
    assert out.shape == (5, 64, 64)
    np.testing.assert_allclose(out, _dfi_per_slice(sinograms, angles, [31.5] * 5), rtol=1e-4, atol=1e-6)


def test_dfi_with_different_centers_matches_per_slice(stack):
    sinograms, angles = stack
    centers = np.array([30.0, 30.5, 31.0, 31.5, 32.0])
    out = get_backend("algotom_dfi").reconstruct(sinograms.copy(), angles, centers, GEOMETRY)
    np.testing.assert_allclose(out, _dfi_per_slice(sinograms, angles, centers), rtol=1e-4, atol=1e-6)


def test_dfi_single_slice_keeps_the_stack_shape(stack):
    sinograms, angles = stack
    assert get_backend("algotom_dfi").reconstruct(sinograms[:, :1].copy(), angles, 31.5, GEOMETRY).shape == (1, 64, 64)


@pytest.mark.parametrize("backend", ["algotom_dfi", "algotom_bpf", "algotom_fbp"])
def test_backends_do_not_modify_a_sinogram_containing_non_positive_values(stack, backend):
    sinograms, angles = stack
    sinograms = sinograms.copy()
    sinograms[3, 2, 10] = 0.0  # algotom would silently overwrite this with 1.0 in the caller's array
    before = sinograms.copy()
    get_backend(backend).reconstruct(sinograms, angles, 31.5, GEOMETRY, device="cpu")
    np.testing.assert_array_equal(sinograms, before)
