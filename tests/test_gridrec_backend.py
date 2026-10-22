"""The `gridrec` reconstruction backend (a wrapper over the urct_gridrec package)."""
import sys

import numpy as np
import pytest

from reconstruction import Geometry, get_backend, list_backends, reconstruct, resolve_device

pytest.importorskip("urct_gridrec")
algotom = pytest.importorskip("algotom")

GEOMETRY = Geometry(geometry_type="parallel")


def disk_stack(width=48, n_angles=90, n_slices=3):
    """Transmission sinograms of a disk phantom (parallel beam), as normalization would give them."""
    from scipy.ndimage import rotate

    y, x = np.mgrid[:width, :width] - (width - 1) / 2
    img = (((x - 3) ** 2 + (y + 2) ** 2) < 12 ** 2) * 1.0 + (((x + 9) ** 2 + (y - 8) ** 2) < 5 ** 2) * 0.5
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    line = np.stack([rotate(img.astype(np.float32), -a, reshape=False, order=1).sum(axis=0) for a in angles])
    trans = np.exp(-line * 0.04).astype(np.float32)
    return np.repeat(trans[:, None, :], n_slices, axis=1), angles


def test_gridrec_is_registered_and_offered():
    assert get_backend("gridrec").name == "gridrec"
    assert "gridrec" in list_backends()
    assert get_backend("gridrec").supported_geometries == {"parallel"}


def test_output_shape_dtype_and_agreement_with_fbp():
    stack, angles = disk_stack()
    gridrec = reconstruct(stack, angles, 23.5, geometry=GEOMETRY, backend="gridrec")
    fbp = reconstruct(stack, angles, 23.5, geometry=GEOMETRY, backend="algotom_fbp", device="cpu")
    assert gridrec.shape == (3, 48, 48) and gridrec.dtype == np.float32
    assert np.corrcoef(gridrec.ravel(), fbp.ravel())[0, 1] > 0.995
    assert gridrec.mean() / fbp.mean() == pytest.approx(1.0, abs=0.05)      # same units and scale


def test_a_single_sinogram_and_per_slice_centers():
    stack, angles = disk_stack(n_slices=2)
    one = reconstruct(stack[:, 0, :], angles, 23.5, geometry=GEOMETRY, backend="gridrec")
    assert one.shape == (48, 48)
    shifted = np.stack([stack[:, 0, :], np.roll(stack[:, 1, :], 2, axis=1)], axis=1)
    both = reconstruct(shifted, angles, np.array([23.5, 25.5]), geometry=GEOMETRY, backend="gridrec")
    alone = reconstruct(shifted[:, 1, :], angles, 25.5, geometry=GEOMETRY, backend="gridrec")
    assert np.linalg.norm(both[1] - alone) / np.linalg.norm(alone) < 3e-4


def test_options_are_passed_through():
    stack, angles = disk_stack(n_slices=1)
    ramp = reconstruct(stack, angles, 23.5, geometry=GEOMETRY, backend="gridrec", filter_name="ramp")
    hann = reconstruct(stack, angles, 23.5, geometry=GEOMETRY, backend="gridrec", filter_name="hann")
    assert not np.allclose(ramp, hann)
    with pytest.raises(TypeError):
        reconstruct(stack, angles, 23.5, geometry=GEOMETRY, backend="gridrec", not_an_option=1)


def test_cone_geometry_is_refused():
    stack, angles = disk_stack(n_slices=1)
    with pytest.raises(ValueError, match="doesn't support"):
        reconstruct(stack, angles, 23.5, geometry=Geometry(geometry_type="cone"), backend="gridrec")


def test_input_is_not_modified():
    stack, angles = disk_stack(n_slices=2)
    stack[5, 0, 3] = 0.0
    before = stack.copy()
    reconstruct(stack, angles, 23.5, geometry=GEOMETRY, backend="gridrec")
    np.testing.assert_array_equal(stack, before)


def test_device_cpu_unless_the_gpu_is_asked_for_and_exists(monkeypatch):
    import urct_gridrec

    assert resolve_device("gridrec", None) == "cpu"       # "auto" is deliberately CPU
    assert resolve_device("gridrec", "cpu") == "cpu"
    monkeypatch.setattr(urct_gridrec, "gpu_available", lambda: False)
    assert resolve_device("gridrec", "gpu") == "cpu"      # asked for, but there is none
    monkeypatch.setattr(urct_gridrec, "gpu_available", lambda: True)
    assert resolve_device("gridrec", "gpu") == "gpu"


def test_missing_package_gives_an_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "urct_gridrec", None)          # makes `import urct_gridrec` fail
    stack, angles = disk_stack(n_slices=1)
    with pytest.raises(ImportError, match="pip install"):
        reconstruct(stack, angles, 23.5, geometry=GEOMETRY, backend="gridrec")


def test_unavailable_when_the_package_is_not_installed(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name == "urct_gridrec" else real(name, *a, **k))
    assert not get_backend("gridrec").is_available()
    assert "gridrec" not in list_backends()
    from reconstruction.backends.gridrec import warm_up

    warm_up()      # a no-op, not an error


def test_warm_up_runs():
    from reconstruction.backends.gridrec import warm_up

    warm_up()


# -- through the whole pipeline ------------------------------------------------------------

def test_pipeline_runs_with_the_gridrec_backend_eager_and_streaming(tmp_path):
    import h5py

    from orchestration import PipelineRun, ProjectConfig
    from tests.fixtures.realistic_phantom import generate as generate_phantom
    from tests.fixtures.realistic_phantom import write_hdf5

    source = tmp_path / "scan.h5"
    write_hdf5(generate_phantom(
        shape=(8, 32, 32), n_angles=90, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=1,
    ), source)

    def run(backend, name, **extra):
        config = ProjectConfig(
            path=str(source), format="hdf5", center=16.0, tilt_deg=0.0, reconstruction_backend=backend,
            export_path=str(tmp_path / name), **extra,
        )
        result = PipelineRun(config).run_all()
        assert result.success, result.error
        with h5py.File(result.volume_path, "r") as f:
            return f["/volume"][:]

    eager = run("gridrec", "eager.h5")
    streamed = run("gridrec", "stream.h5", stream=True, stream_chunk_rows=3)
    fbp = run("algotom_fbp", "fbp.h5")
    assert eager.shape == streamed.shape == fbp.shape == (8, 32, 32)
    # Chunking changes which slices share a transform; that moves the result by far less than a percent.
    assert np.linalg.norm(streamed - eager) / np.linalg.norm(eager) < 3e-3
    assert np.corrcoef(eager.ravel(), fbp.ravel())[0, 1] > 0.99
