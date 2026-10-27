"""
End-to-end: ILL-style folder -> ill_to_hdf5 -> tomo CLI / PipelineRun with
projections_normalized and a 360 degree angle span.
"""
import h5py
import numpy as np
import pytest
import tifffile

import ill_to_hdf5 as conv
from calibration.backend import CalibrationBackend, register_backend
from orchestration.cli import main
from orchestration.config import ProjectConfig
from orchestration.pipeline import PipelineRun

pytest.importorskip("algotom", reason="algotom not installed -- these tests need the algotom_fbp backend")

N_ANGLES = 40


@pytest.fixture
def converted(tmp_path):
    folder = tmp_path / "ill"
    (folder / "Proj").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(N_ANGLES):
        tifffile.imwrite(folder / "Proj" / f"proj_{i:04d}.tif", rng.integers(500, 900, (16, 32), dtype=np.uint16))
    junk = np.full((16, 32), 5, np.uint16)  # flat/dark that must be ignored
    tifffile.imwrite(folder / "FlatsNeutrons.tif", junk)
    tifffile.imwrite(folder / "DarksNeutrons.tif", junk)
    (folder / "scan_information.xml").write_text('<scanInformation><imager pixelSize_mm="0.02"/></scanInformation>')
    out = tmp_path / "ill.h5"
    conv.convert(folder, out, quiet=True)
    return out


def test_tomo_run_with_projections_normalized_and_angle_span(converted, tmp_path, capsys):
    config = tmp_path / "scan.ini"
    out = tmp_path / "volume.h5"
    config.write_text(
        f"[import]\npath = {converted}\nformat = hdf5\n\n"
        "[completion]\nangle_span = 360\n\n"
        "[preprocessing]\nprojections_normalized = true\n\n"
        "[calibration]\ncenter = 16\ntilt_deg = 0\n\n"
        f"[export]\nformat = hdf5\npath = {out}\n"
    )

    assert main(["run", str(config)]) == 0
    with h5py.File(out, "r") as f:
        assert f["/entry/volume/data"].shape == (16, 32, 32)
        assert np.isfinite(f["/entry/volume/data"][()]).all()


def test_tomo_validate_accepts_the_converted_file(converted, tmp_path):
    config = tmp_path / "scan.ini"
    config.write_text(
        f"[import]\npath = {converted}\nformat = hdf5\n\n"
        "[completion]\nangle_span = 360\n\n[preprocessing]\nprojections_normalized = true\n\n"
        "[calibration]\ncenter = 16\ntilt_deg = 0\n"
    )
    assert main(["validate", str(config)]) == 0


def test_calibration_sees_only_the_first_180_degrees_of_a_360_scan(converted):
    seen = []

    @register_backend
    class _Spy(CalibrationBackend):
        name = "spy_half_scan"

        def find_center(self, sino_180, start=None, stop=None, **options):
            seen.append(sino_180.shape[0])
            return sino_180.shape[-1] / 2.0

    config = ProjectConfig(
        path=str(converted), format="hdf5", completion_answers={"angle_span": 360.0},
        projections_normalized=True, calibration_backend="spy_half_scan",
    )
    run = PipelineRun(config)
    run.import_data()
    run.complete()
    run.preprocess()
    run.calibrate()

    assert seen and set(seen) == {N_ANGLES // 2}


def test_calibration_uses_every_angle_for_a_180_degree_scan(converted):
    seen = []

    @register_backend
    class _Spy180(CalibrationBackend):
        name = "spy_full_scan"

        def find_center(self, sino_180, start=None, stop=None, **options):
            seen.append(sino_180.shape[0])
            return sino_180.shape[-1] / 2.0

    config = ProjectConfig(
        path=str(converted), format="hdf5", completion_answers={"angle_span": 180.0},
        projections_normalized=True, calibration_backend="spy_full_scan",
    )
    run = PipelineRun(config)
    run.import_data()
    run.complete()
    run.preprocess()
    run.calibrate()

    assert set(seen) == {N_ANGLES}
