"""Cone-beam in the GUI: the distance fields, the geometry-dependent algorithm list, and a full reconstruction."""
import os
import time

import h5py
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
pytest.importorskip("itk")
from PySide6.QtWidgets import QApplication  # noqa: E402

import gui.reconstruct_dialog as reconstruct_dialog  # noqa: E402
from gui.adjustment_panel import AdjustmentPanel, BasicSettingsTab  # noqa: E402
from gui.session import ProjectSession  # noqa: E402
from gui.viewer import ImageViewer  # noqa: E402
from orchestration import PipelineRun, ProjectConfig  # noqa: E402
from reconstruction import list_backends  # noqa: E402
from tests.test_cone_beam import (  # noqa: E402
    AXIS_COLUMN, COLS, N_ANGLES, ROWS, SDD, SOD, SPHERES, analytic_projections, blobs, voxel_size_mm,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def cone_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("cone") / "cone.h5"
    angles = np.arange(N_ANGLES) * 360.0 / N_ANGLES
    with h5py.File(path, "w") as f:
        f.attrs["beamline"] = "generic_parallel_synchrotron"
        f.create_dataset("/entry/instrument/detector/data", data=np.exp(-analytic_projections()).astype(np.float32))
        f.create_dataset("/entry/instrument/detector/flat_fields", data=np.ones((2, ROWS, COLS), dtype=np.float32))
        f.create_dataset("/entry/instrument/detector/dark_fields", data=np.zeros((2, ROWS, COLS), dtype=np.float32))
        f.create_dataset("/entry/instrument/sample/rotation_angle", data=angles)
        f.create_dataset("/entry/instrument/detector/pixel_size", data=voxel_size_mm() * 1000.0)
    return path


def _session(path, **answers):
    config = ProjectConfig(path=str(path), format="hdf5", storage_mode="disk", center=AXIS_COLUMN, tilt_deg=0.0,
                           completion_answers={"geometry_type": "parallel", **answers})
    run = PipelineRun(config)
    run.import_data()
    run.complete()
    return ProjectSession(config=config, run=run)


# -- the tab alone ----------------------------------------------------------------------------------

def test_distance_fields_appear_only_for_cone_beam(app):
    tab = BasicSettingsTab()
    args = dict(angle_span=180.0, angle_span_derived=False, pixel_size=1.0, pixel_size_derived=False, slice_index=0,
                slice_max=10, backend="algotom_fbp", num_iter=50, is_iterative=False, device=None, center=10.0,
                tilt_deg=0.0, rotate_deg=0.0, width=20)
    tab.set_values(geometry_type="parallel", **args)
    assert all(spin.isHidden() for spin in tab.distance_spins.values()) and tab.sdd_label.isHidden()
    tab.set_values(geometry_type="cone", distances={"source_to_detector_distance": 300.0}, **{**args, "backend": "rtk_fdk"})
    assert not any(spin.isHidden() for spin in tab.distance_spins.values())
    assert tab.distance_spins["source_to_detector_distance"].value() == 300.0
    assert tab.distance_spins["source_to_object_distance"].text() == "not set"


def test_editing_a_distance_emits_its_field_name(app):
    tab = BasicSettingsTab()
    seen = []
    tab.distance_changed.connect(lambda key, value: seen.append((key, value)))
    tab.distance_spins["source_to_object_distance"].setValue(150.0)
    assert seen == [("source_to_object_distance", 150.0)]


# -- the panel --------------------------------------------------------------------------------------

def test_switching_geometry_changes_the_algorithm_list_and_the_backend(app, cone_file):
    session = _session(cone_file)
    try:
        panel = AdjustmentPanel(ImageViewer())
        panel.set_session(session)
        tab = panel.basic_tab
        items = lambda: [tab.backend_combo.itemText(i) for i in range(tab.backend_combo.count())]
        assert items() == list_backends(available_only=False, geometry="parallel") and session.config.reconstruction_backend == "algotom_fbp"

        tab.geometry_combo.setCurrentText("cone")
        assert items() == list_backends(available_only=False, geometry="cone") and session.config.reconstruction_backend == "rtk_fdk"
        assert session.config.completion_answers["geometry_type"] == "cone"

        tab.distance_spins["source_to_detector_distance"].setValue(300.0)
        tab.distance_spins["source_to_object_distance"].setValue(150.0)
        assert session.config.completion_answers["source_to_detector_distance"] == 300.0
        assert session.config.completion_answers["source_to_object_distance"] == 150.0
        tab.distance_spins["source_to_object_distance"].setValue(0.0)         # "not set" removes the answer again
        assert "source_to_object_distance" not in session.config.completion_answers

        tab.geometry_combo.setCurrentText("parallel")
        assert items() == list_backends(available_only=False, geometry="parallel") and session.config.reconstruction_backend == "algotom_fbp"
    finally:
        session.run.project.close()


def test_a_cone_project_reconstructs_from_the_panel(app, cone_file):
    session = _session(
        cone_file, geometry_type="cone", source_to_detector_distance=SDD, source_to_object_distance=SOD,
    )
    try:
        panel = AdjustmentPanel(ImageViewer())
        panel.set_session(session)
        assert session.config.reconstruction_backend == "rtk_fdk"            # chosen because the geometry is cone
        reconstruct_dialog.ReconstructDestinationDialog.exec = lambda self: (self.ram_radio.setChecked(True) or True)
        panel._reconstruct(all_slices=True)
        deadline = time.time() + 120
        while session.ram_volume is None and time.time() < deadline:
            app.processEvents()
            time.sleep(0.05)
        assert session.ram_volume is not None
        assert session.ram_volume.shape == (ROWS, COLS, COLS)
        found = blobs(session.ram_volume)
        assert len(found) == 2 and found[0][:3] == pytest.approx(SPHERES[0][:3], abs=0.6)
    finally:
        session.run.project.close()


def test_a_single_slice_preview_explains_itself_for_cone_beam(app, cone_file):
    session = _session(
        cone_file, geometry_type="cone", source_to_detector_distance=SDD, source_to_object_distance=SOD,
    )
    try:
        panel = AdjustmentPanel(ImageViewer())
        panel.set_session(session)
        messages = []
        panel.status_message.connect(messages.append)
        panel._reconstruct(all_slices=False)
        deadline = time.time() + 60
        while not any("failed" in m for m in messages) and time.time() < deadline:
            app.processEvents()
            time.sleep(0.05)
        assert any("Reconstruct all slices" in m for m in messages)
    finally:
        session.run.project.close()
