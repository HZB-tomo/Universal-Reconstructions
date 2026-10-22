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


class _Mem:
    available = 64 * 1024**3


def test_gpu_chunks_hold_enough_rows_to_keep_the_filter_pipeline_full(monkeypatch):
    monkeypatch.setattr(ram_check.psutil, "virtual_memory", lambda: _Mem())
    monkeypatch.setattr(ram_check.os, "cpu_count", lambda: 8)
    assert ram_check.estimate_chunk_rows(1024, 5000, "cpu", 1000) == 8
    assert ram_check.estimate_chunk_rows(1024, 5000, "gpu", 1000) == ram_check.GPU_CHUNK_ROWS
    assert ram_check.estimate_chunk_rows(1024, 5, "gpu", 1000) == 5  # never more rows than exist


def test_gpu_chunks_still_respect_free_ram(monkeypatch):
    class Tight:
        available = int(ram_check.estimate_volume_bytes(1, 1024) * ram_check.SAFETY_FACTOR * 3.5)

    monkeypatch.setattr(ram_check.psutil, "virtual_memory", lambda: Tight())
    monkeypatch.setattr(ram_check.os, "cpu_count", lambda: 8)
    assert ram_check.estimate_chunk_rows(1024, 5000, "gpu", 1000) == 3


# -- the GPU path, run on numba's CUDA simulator ---------------------------------------
# No GPU is needed: NUMBA_ENABLE_CUDASIM=1 executes the real kernels in Python. It must be
# set before numba is imported, hence the subprocess. This checks the GPU code's results
# and that the many-slices-per-launch kernel is never used; it says nothing about speed.

_SIM_SCRIPT = r"""
import os, sys, warnings
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
import numpy as np
from algotom.rec import reconstruction as rec
assert rec.cuda.is_available()

def forbidden(*a, **k):
    raise AssertionError("the many-slices-per-launch GPU kernel must not be used")
rec.back_projection_gpu_chunk = forbidden
rec.__dict__["__back_projection_gpu_chunk_kernel"] = forbidden

from reconstruction import Geometry, get_backend
from reconstruction.backends._fbp_cpu import bpf_stack, fbp_stack

rng = np.random.default_rng(0)
stack = rng.uniform(0.2, 1.0, (24, 4, 32)).astype(np.float32)
stack[3, 1, 5] = 0.0                      # a non-positive value algotom would overwrite in place
angles = np.linspace(0, 180, 24, endpoint=False)
before = stack.copy()
geo = Geometry(geometry_type="parallel")
inner = (slice(None), slice(1, -1), slice(1, -1))
for name, cpu_fn in (("algotom_fbp", fbp_stack), ("algotom_bpf", bpf_stack)):
    backend = get_backend(name)
    assert backend.actual_device("gpu") == "gpu"
    for centers in (15.5, np.array([15.0, 15.5, 16.0, 16.5])):       # shared center, then tilt-like
        gpu = backend.reconstruct(stack, angles, centers, geo, device="gpu")
        cpu = cpu_fn(rec, stack, np.broadcast_to(np.asarray(centers, float), (4,)), np.radians(angles))
        err = np.linalg.norm(gpu[inner] - cpu[inner]) / np.linalg.norm(cpu[inner])
        assert gpu.shape == (4, 32, 32) and gpu.dtype == np.float32, (name, gpu.shape, gpu.dtype)
        assert err < 1e-4, (name, err)
assert np.array_equal(stack, before), "input was modified"
print("OK")
"""


def test_gpu_fbp_and_bpf_match_the_cpu_results_on_the_cuda_simulator():
    import subprocess
    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, "-c", _SIM_SCRIPT, root], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0 and "OK" in result.stdout, result.stdout + result.stderr


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
