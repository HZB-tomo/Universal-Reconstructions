"""Cone-beam FDK (the rtk_fdk backend): geometry conventions against analytic phantoms, errors, pipeline guards."""
import sys

import h5py
import numpy as np
import pytest
from scipy import ndimage as ndi

from orchestration import PipelineRun, PipelineStageError, ProjectConfig
from orchestration.geometry import geometry_from_project
from reconstruction import Geometry, get_backend, list_backends, reconstruct, resolve_device

pytest.importorskip("itk")

SOD, SDD, PIXEL_MM = 150.0, 300.0, 1.0
ROWS, COLS, N_ANGLES = 48, 64, 120
AXIS_COLUMN = (COLS - 1) / 2.0 + 2.0              # the rotation axis projects 2 pixels off the detector middle
SPHERES = [                                      # x, y, z (mm; z along the rotation axis), radius, mu (1/mm)
    (10.0, -5.0, 7.5, 4.0, 0.05),
    (-12.0, 6.0, -2.5, 2.5, 0.08),
]
CONE = Geometry(geometry_type="cone", source_to_detector_distance=SDD, source_to_object_distance=SOD,
                detector_pixel_size=PIXEL_MM * 1000.0)


def analytic_projections(axis_column=AXIS_COLUMN):
    """Line integrals of the spheres for the circular cone-beam scan described in rtk_fdk's docstring:
    detector axis u = (cos t, -sin t), ray r = (sin t, cos t), source at -SOD * r."""
    cc, rr = np.meshgrid(np.arange(COLS), np.arange(ROWS))
    zhat = np.array([0.0, 0.0, 1.0])
    out = np.empty((N_ANGLES, ROWS, COLS), dtype=np.float32)
    for k, theta in enumerate(np.radians(np.arange(N_ANGLES) * 360.0 / N_ANGLES)):
        u = np.array([np.cos(theta), -np.sin(theta), 0.0])
        r = np.array([np.sin(theta), np.cos(theta), 0.0])
        source = -SOD * r
        detector_centre = (SDD - SOD) * r
        point = (detector_centre[None, None, :] + ((cc - axis_column) * PIXEL_MM)[..., None] * u
                 + ((rr - (ROWS - 1) / 2) * PIXEL_MM)[..., None] * zhat)
        ray = point - source
        length = np.linalg.norm(ray, axis=-1)
        total = np.zeros((ROWS, COLS))
        for x0, y0, z0, radius, mu in SPHERES:
            d = np.linalg.norm(np.cross((np.array([x0, y0, z0]) - source)[None, None, :], ray), axis=-1) / length
            total += mu * 2 * np.sqrt(np.clip(radius ** 2 - d ** 2, 0, None))
        out[k] = total
    return out


@pytest.fixture(scope="module")
def phantom():
    angles = np.arange(N_ANGLES) * 360.0 / N_ANGLES
    transmission = np.exp(-analytic_projections()).astype(np.float32)
    volume = reconstruct(transmission, angles, AXIS_COLUMN, geometry=CONE, backend="rtk_fdk")
    return transmission, angles, volume


def voxel_size_mm():
    return PIXEL_MM * SOD / SDD


def blobs(volume, threshold_fraction=0.4):
    """Centroids, in mm in the scan's (x, y, z) frame, of the connected blobs above a threshold."""
    labelled, count = ndi.label(volume > threshold_fraction * volume.max())
    found = []
    for label in range(1, count + 1):
        mask = labelled == label
        if mask.sum() < 20:
            continue
        index = np.argwhere(mask)
        weight = volume[mask]
        iz, iy, ix = (index * weight[:, None]).sum(0) / weight.sum()
        vox = voxel_size_mm()
        nz, ny, nx = volume.shape
        found.append(((ix - (nx - 1) / 2) * vox, (iy - (ny - 1) / 2) * vox, (iz - (nz - 1) / 2) * vox, int(mask.sum())))
    return sorted(found, key=lambda b: -b[3])


def test_backend_is_registered_for_cone_beam_only():
    backend = get_backend("rtk_fdk")
    assert backend.supported_geometries == {"cone"} and backend.is_available()
    assert list_backends(geometry="cone") == ["rtk_fdk"]
    assert "rtk_fdk" not in list_backends(geometry="parallel")
    assert resolve_device("rtk_fdk", "gpu") == "cpu"          # the CPU implementation; says so honestly


def test_output_shape_and_dtype(phantom):
    _, _, volume = phantom
    assert volume.shape == (ROWS, COLS, COLS) and volume.dtype == np.float32      # (slices along the axis, y, x)


def test_the_spheres_are_where_they_should_be_on_all_three_axes(phantom):
    """The geometry convention end to end: positions of two asymmetric spheres within a voxel."""
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
    background = np.abs(volume[:5]).mean()
    assert background < 0.05 * mu * 10.0


def test_a_wrong_center_visibly_degrades_the_result(phantom):
    """The center argument matters: 6 pixels off smears every feature (a negative control for the test above)."""
    transmission, angles, volume = phantom
    wrong = reconstruct(transmission, angles, AXIS_COLUMN + 6.0, geometry=CONE, backend="rtk_fdk")
    assert np.linalg.norm(wrong - volume) / np.linalg.norm(volume) > 0.3
    assert wrong.max() < 0.85 * volume.max()


def test_filters_and_sizes(phantom):
    transmission, angles, _ = phantom
    ramp = reconstruct(transmission, angles, AXIS_COLUMN, geometry=CONE, backend="rtk_fdk", filter_name="ramp", size_xy=48, size_z=40)
    assert ramp.shape == (40, 48, 48)


def test_errors_are_clear(phantom):
    transmission, angles, _ = phantom
    backend = get_backend("rtk_fdk")
    with pytest.raises(ValueError, match="needs source_to_object_distance"):
        backend.reconstruct(transmission, angles, AXIS_COLUMN, Geometry(geometry_type="cone", source_to_detector_distance=SDD, detector_pixel_size=1000.0))
    with pytest.raises(ValueError, match="source_to_detector_distance"):
        backend.reconstruct(transmission, angles, AXIS_COLUMN, Geometry("cone", 100.0, 200.0, 1000.0))
    with pytest.raises(ValueError, match="one center"):
        backend.reconstruct(transmission, angles, np.linspace(30, 34, ROWS), CONE)
    with pytest.raises(ValueError, match="unknown filter"):
        backend.reconstruct(transmission, angles, AXIS_COLUMN, CONE, filter_name="nope")
    with pytest.raises(TypeError, match="unexpected options"):
        backend.reconstruct(transmission, angles, AXIS_COLUMN, CONE, not_an_option=1)
    with pytest.raises(ValueError, match="doesn't support"):
        reconstruct(transmission, angles, AXIS_COLUMN, geometry=Geometry(geometry_type="parallel"), backend="rtk_fdk")


def test_missing_package_gives_an_install_hint(phantom, monkeypatch):
    transmission, angles, _ = phantom
    monkeypatch.setitem(sys.modules, "itk", None)
    with pytest.raises(ImportError, match="itk-rtk"):
        get_backend("rtk_fdk").reconstruct(transmission, angles, AXIS_COLUMN, CONE)


# -- geometry from a project --------------------------------------------------------------------

@pytest.fixture
def cone_scan(tmp_path, phantom):
    transmission, angles, _ = phantom
    path = tmp_path / "cone.h5"
    with h5py.File(path, "w") as f:
        f.attrs["beamline"] = "generic_parallel_synchrotron"
        f.create_dataset("/entry/instrument/detector/data", data=transmission)
        f.create_dataset("/entry/instrument/detector/flat_fields", data=np.ones((2, ROWS, COLS), dtype=np.float32))
        f.create_dataset("/entry/instrument/detector/dark_fields", data=np.zeros((2, ROWS, COLS), dtype=np.float32))
        f.create_dataset("/entry/instrument/sample/rotation_angle", data=angles)
        f.create_dataset("/entry/instrument/detector/pixel_size", data=voxel_size_mm() * 1000.0)     # pixel size at the object, um
    return path


def _cone_config(path, **kwargs):
    answers = {"geometry_type": "cone", "source_to_detector_distance": SDD, "source_to_object_distance": SOD}
    base = dict(path=str(path), format="hdf5", storage_mode="disk", center=AXIS_COLUMN, tilt_deg=0.0,
                reconstruction_backend="rtk_fdk", completion_answers=answers)
    return ProjectConfig(**{**base, **kwargs})


def _imported(path, **kwargs):
    config = _cone_config(path, **kwargs)
    run = PipelineRun(config)
    run.import_data()
    run.complete()
    return run, config


def test_detector_pixel_size_of_a_cone_project_is_the_voxel_size_times_the_magnification(cone_scan):
    run, config = _imported(cone_scan)
    geometry = geometry_from_project(run.project)
    assert geometry.geometry_type == "cone"
    assert geometry.detector_pixel_size == pytest.approx(PIXEL_MM * 1000.0)       # 500 um at the object * 2
    run.project.close()


def test_a_detector_pitch_given_explicitly_wins(cone_scan):
    config = _cone_config(cone_scan)
    config.completion_answers["detector_pixel_size"] = 750.0
    run = PipelineRun(config)
    run.import_data()
    run.complete()
    assert geometry_from_project(run.project).detector_pixel_size == 750.0
    run.project.close()


def test_a_parallel_project_keeps_its_detector_pixel_size(tmp_path):
    from tests.fixtures.realistic_phantom import generate as generate_phantom
    from tests.fixtures.realistic_phantom import write_hdf5

    path = tmp_path / "p.h5"
    write_hdf5(generate_phantom(shape=(4, 16, 16), n_angles=20, seed=1), path)
    run = PipelineRun(ProjectConfig(path=str(path), format="hdf5", completion_answers={"geometry_type": "parallel"}))
    run.import_data()
    run.complete()
    assert geometry_from_project(run.project).detector_pixel_size == run.project.metadata.get("detector_pixel_size")
    run.project.close()


# -- through the pipeline -----------------------------------------------------------------------

def test_pipeline_reconstructs_a_cone_beam_project(cone_scan):
    config = _cone_config(cone_scan, export_path=str(cone_scan.parent / "out.h5"))
    result = PipelineRun(config).run_all()
    assert result.success, result.error
    with h5py.File(result.volume_path, "r") as f:
        volume = f["/entry/volume/data"][:]
    assert volume.shape == (ROWS, COLS, COLS)
    found = blobs(volume)
    assert len(found) == 2
    assert found[0][:3] == pytest.approx(SPHERES[0][:3], abs=0.6)


def test_the_pipeline_refuses_what_cone_beam_cannot_do(cone_scan):
    run, config = _imported(cone_scan, rows=slice(10, 20))
    run.preprocess(); run.calibrate()
    with pytest.raises(PipelineStageError, match="every detector row"):
        run.reconstruct()
    run.project.close()

    tilted = _cone_config(cone_scan, tilt_deg=0.5)
    result = PipelineRun(tilted).run_all()
    assert not result.success and "tilt" in str(result.error)

    streamed = _cone_config(cone_scan, stream=True, center=AXIS_COLUMN, export_path=str(cone_scan.parent / "s.h5"))
    result = PipelineRun(streamed).run_all()
    assert not result.success and "streaming" in str(result.error)


def test_single_slice_previews_say_why_they_are_unavailable(cone_scan):
    from gui.preview_engine import PreviewEngine

    run, config = _imported(cone_scan)
    with pytest.raises(ValueError, match="Reconstruct all slices"):
        PreviewEngine().compute(run.project, config, 10, AXIS_COLUMN)
    run.project.close()
