"""Cone-beam FDK on the GPU (the astra_fdk backend): same conventions as rtk_fdk, against analytic phantoms."""
import numpy as np
import pytest

from reconstruction import Geometry, get_backend, list_backends, reconstruct, resolve_device

astra = pytest.importorskip("astra")
if not astra.use_cuda():
    pytest.skip("ASTRA sees no CUDA device", allow_module_level=True)

from tests.fixtures.cone_phantom import (  # noqa: E402
    AXIS_COLUMN, CONE, COLS, N_ANGLES, PIXEL_MM, ROWS, SDD, SOD, SPHERES, analytic_projections, blobs, voxel_size_mm,
)


@pytest.fixture(scope="module")
def phantom():
    angles = np.arange(N_ANGLES) * 360.0 / N_ANGLES
    transmission = np.exp(-analytic_projections()).astype(np.float32)
    volume = reconstruct(transmission, angles, AXIS_COLUMN, geometry=CONE, backend="astra_fdk")
    return transmission, angles, volume


def test_backend_is_registered_for_cone_beam_on_the_gpu():
    backend = get_backend("astra_fdk")
    assert backend.supported_geometries == {"cone"} and backend.is_available() and backend.unavailable_reason() == ""
    assert "astra_fdk" in list_backends(geometry="cone") and "astra_fdk" not in list_backends(geometry="parallel")
    assert resolve_device("astra_fdk", "gpu") == "gpu" and resolve_device("astra_fdk", "cpu") == "gpu"


def test_unavailable_without_a_gpu(monkeypatch):
    monkeypatch.setattr(astra, "use_cuda", lambda: False)
    backend = get_backend("astra_fdk")
    assert not backend.is_available() and "GPU" in backend.unavailable_reason() and "rtk_fdk" in backend.unavailable_reason()
    assert "astra_fdk" not in list_backends(geometry="cone")
    assert "astra_fdk" in list_backends(available_only=False, geometry="cone")


def test_output_shape_and_dtype(phantom):
    _, _, volume = phantom
    assert volume.shape == (ROWS, COLS, COLS) and volume.dtype == np.float32      # (slices along the axis, y, x)


def test_the_spheres_are_where_they_should_be_on_all_three_axes(phantom):
    _, _, volume = phantom
    found = blobs(volume)
    assert len(found) == 2
    for (x, y, z, _), (x0, y0, z0, _radius, _mu) in zip(found, SPHERES):
        assert (x, y, z) == pytest.approx((x0, y0, z0), abs=0.6)       # a voxel is 0.5 mm


def test_attenuation_values_come_back_in_per_cm(phantom):
    _, _, volume = phantom
    vox = voxel_size_mm()
    nz, ny, nx = volume.shape
    zz, yy, xx = np.mgrid[:nz, :ny, :nx]
    x0, y0, z0, radius, mu = SPHERES[0]
    inside = (((xx - (nx - 1) / 2) * vox - x0) ** 2 + ((yy - (ny - 1) / 2) * vox - y0) ** 2
              + ((zz - (nz - 1) / 2) * vox - z0) ** 2) < (0.6 * radius) ** 2
    assert inside.sum() > 50
    assert volume[inside].mean() == pytest.approx(mu * 10.0, rel=0.1)       # 1/mm -> 1/cm
    assert np.abs(volume[:5]).mean() < 0.05 * mu * 10.0


def test_agrees_with_the_cpu_backend(phantom):
    pytest.importorskip("itk")
    transmission, angles, volume = phantom
    cpu = reconstruct(transmission, angles, AXIS_COLUMN, geometry=CONE, backend="rtk_fdk")
    assert np.linalg.norm(volume - cpu) / np.linalg.norm(cpu) < 0.1


def test_a_wrong_center_visibly_degrades_the_result(phantom):
    transmission, angles, volume = phantom
    wrong = reconstruct(transmission, angles, AXIS_COLUMN + 6.0, geometry=CONE, backend="astra_fdk")
    assert np.linalg.norm(wrong - volume) / np.linalg.norm(volume) > 0.3
    assert wrong.max() < 0.85 * volume.max()


def test_filters_cut_frequency_and_sizes(phantom):
    transmission, angles, volume = phantom
    ramp = reconstruct(transmission, angles, AXIS_COLUMN, geometry=CONE, backend="astra_fdk", filter_name="ramp", size_xy=48, size_z=40)
    assert ramp.shape == (40, 48, 48)
    soft = reconstruct(transmission, angles, AXIS_COLUMN, geometry=CONE, backend="astra_fdk", cut_frequency=0.4)
    assert soft.shape == volume.shape and not np.allclose(soft, volume)
    assert np.abs(np.diff(soft, axis=2)).mean() < np.abs(np.diff(volume, axis=2)).mean()      # a lower cutoff smooths


def test_errors_are_clear(phantom):
    transmission, angles, _ = phantom
    backend = get_backend("astra_fdk")
    with pytest.raises(ValueError, match="needs source_to_object_distance"):
        backend.reconstruct(transmission, angles, AXIS_COLUMN, Geometry(geometry_type="cone", source_to_detector_distance=SDD, detector_pixel_size=1000.0))
    with pytest.raises(ValueError, match="source_to_detector_distance"):
        backend.reconstruct(transmission, angles, AXIS_COLUMN, Geometry("cone", 100.0, 200.0, 1000.0))
    with pytest.raises(ValueError, match="one center"):
        backend.reconstruct(transmission, angles, np.linspace(30, 34, ROWS), CONE)
    with pytest.raises(ValueError, match="needs the center of rotation"):
        backend.reconstruct(transmission, angles, None, CONE)
    with pytest.raises(ValueError, match="unknown filter"):
        backend.reconstruct(transmission, angles, AXIS_COLUMN, CONE, filter_name="nope")
    with pytest.raises(TypeError, match="unexpected options"):
        backend.reconstruct(transmission, angles, AXIS_COLUMN, CONE, not_an_option=1)
