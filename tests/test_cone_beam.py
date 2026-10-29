"""Cone-beam FDK (the rtk_fdk backend): geometry conventions against analytic phantoms, errors, pipeline guards."""
import sys

import h5py
import numpy as np
import pytest

from orchestration import PipelineRun, PipelineStageError, ProjectConfig
from orchestration.geometry import geometry_from_project
from reconstruction import Geometry, get_backend, list_backends, reconstruct, resolve_device

pytest.importorskip("itk")

from tests.fixtures.cone_phantom import (  # noqa: E402
    AXIS_COLUMN, CENTRAL_SPHERES, CONE, COLS, N_ANGLES, PIXEL_MM, ROWS, SDD, SOD, SPHERES, analytic_projections, blobs, voxel_size_mm,
)


@pytest.fixture(scope="module")
def phantom():
    angles = np.arange(N_ANGLES) * 360.0 / N_ANGLES
    transmission = np.exp(-analytic_projections()).astype(np.float32)
    volume = reconstruct(transmission, angles, AXIS_COLUMN, geometry=CONE, backend="rtk_fdk")
    return transmission, angles, volume


def test_backend_is_registered_for_cone_beam_only():
    backend = get_backend("rtk_fdk")
    assert backend.supported_geometries == {"cone"} and backend.is_available()
    assert "rtk_fdk" in list_backends(geometry="cone")
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


@pytest.fixture(scope="module")
def central():
    """A scan whose spheres lie on the plane through the source: transmission, angles and the full volume."""
    angles = np.arange(N_ANGLES) * 360.0 / N_ANGLES
    transmission = np.exp(-analytic_projections(spheres=CENTRAL_SPHERES)).astype(np.float32)
    return transmission, angles, reconstruct(transmission, angles, AXIS_COLUMN, geometry=CONE, backend="rtk_fdk")


def test_the_central_plane_can_be_previewed_on_its_own(central, tmp_path):
    """The plane through the source needs only the middle detector row(s): the preview must equal the middle of the full volume."""
    from gui.preview_engine import PreviewEngine

    transmission, angles, volume = central
    path = tmp_path / "central.h5"
    with h5py.File(path, "w") as f:
        f.attrs["beamline"] = "generic_parallel_synchrotron"
        f.create_dataset("/entry/instrument/detector/data", data=transmission)
        f.create_dataset("/entry/instrument/detector/flat_fields", data=np.ones((2, ROWS, COLS), dtype=np.float32))
        f.create_dataset("/entry/instrument/detector/dark_fields", data=np.zeros((2, ROWS, COLS), dtype=np.float32))
        f.create_dataset("/entry/instrument/sample/rotation_angle", data=angles)
        f.create_dataset("/entry/instrument/detector/pixel_size", data=voxel_size_mm() * 1000.0)
    run, config = _imported(path)
    try:
        engine = PreviewEngine()
        image = engine.compute(run.project, config, 10, AXIS_COLUMN)
        assert image.shape == (COLS, COLS)
        middle = volume[ROWS // 2 - 1:ROWS // 2 + 1].mean(axis=0)       # 48 rows: the plane lies between slices 23 and 24
        assert np.linalg.norm(image - middle) / np.linalg.norm(middle) < 0.05
        assert engine.compute(run.project, config, 30, AXIS_COLUMN) is image          # whatever the row: it is the central plane
        assert engine.reconstructions == 1
    finally:
        run.project.close()


def test_one_detector_row_reconstructs_the_slice_at_the_height_of_the_source(central):
    transmission, angles, volume = central
    row = transmission[:, ROWS // 2:ROWS // 2 + 1, :]        # a single row (half a pixel off the plane)
    one = reconstruct(row, angles, AXIS_COLUMN, geometry=CONE, backend="rtk_fdk")
    assert one.shape == (1, COLS, COLS)
    reference = volume[ROWS // 2 - 1:ROWS // 2 + 1].mean(axis=0)
    assert np.linalg.norm(one[0] - reference) / np.linalg.norm(reference) < 0.1
