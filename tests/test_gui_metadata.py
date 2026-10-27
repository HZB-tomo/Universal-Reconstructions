"""The metadata toolbox: the Qt-free view of a project's metadata, the widget, and its place in the main window."""
import os

import h5py
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6.QtWidgets import QApplication  # noqa: E402

from core import urct_format as fmt  # noqa: E402
from gui.metadata_info import metadata_view  # noqa: E402
from gui.metadata_toolbox import MetadataToolbox  # noqa: E402
from gui.session import ProjectSession  # noqa: E402
from orchestration import PipelineRun, PipelineStageError, ProjectConfig  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def make_file(path, **fields):
    with h5py.File(path, "w") as f:
        fmt.write_header(f, fmt.FILE_PROJECTIONS)
        f.create_dataset(fmt.PROJECTIONS, data=np.ones((6, 4, 5), dtype=np.float32))
        for key, value in fields.items():
            fmt.write_field(f, key, value)
    return path


def make_session(path, **answers):
    config = ProjectConfig(path=str(path), format="hdf5", storage_mode="disk", completion_answers=answers)
    run = PipelineRun(config)
    run.import_data()
    session = ProjectSession(config=config, run=run)
    try:                                      # like MainWindow._import_session: an incomplete project stays open
        run.complete()
    except PipelineStageError as e:
        session.error = str(e)
    return session


def rows_by_key(view):
    return {row.key: row for row in view.rows}


def test_nothing_selected_and_not_imported_yet_say_so():
    assert "Select a project" in metadata_view(None).summary
    pending = ProjectSession(config=ProjectConfig(path="x.h5"))
    assert "still importing" in metadata_view(pending).summary
    failed = ProjectSession(config=ProjectConfig(path="x.h5"), error="boom")
    assert "failed to import: boom" in metadata_view(failed).summary


def test_every_tier_is_listed_with_what_was_read_and_what_is_missing(tmp_path):
    path = make_file(tmp_path / "a.h5", geometry_type="parallel", angles=np.linspace(0, 180, 6, endpoint=False),
                     voxel_size=2.0, energy=18.0)
    session = make_session(path)
    try:
        view = metadata_view(session)
        rows = rows_by_key(view)
        assert rows["geometry_type"].status == "read" and rows["geometry_type"].value == "parallel"
        assert rows["voxel_size"].value == "2" and rows["voxel_size"].unit == "um"
        assert rows["angles"].value.startswith("array (6,)")
        assert rows["energy"].group == "optional" and rows["energy"].status == "read"
        assert rows["operator"].status == "missing" and "enter it" in rows["operator"].note
        assert rows["center_of_rotation"].group == "computed" and rows["center_of_rotation"].status == "later"
        assert rows["source_to_detector_distance"].status == "not_needed"          # parallel beam
        assert rows["projections_path"].group == "mandatory_hard" and rows["projections_path"].status == "read"
        assert {r.group for r in view.rows} >= {"mandatory_hard", "mandatory_soft", "optional", "computed"}
        assert "missing" in view.summary and "read from the file" in view.summary
    finally:
        session.run.project.close()


def test_who_supplied_a_value_is_shown(tmp_path):
    path = make_file(tmp_path / "b.h5", geometry_type="parallel", angles=np.linspace(0, 180, 6, endpoint=False))
    session = make_session(path, sample_name="granite")
    session.config.center = 12.5
    try:
        rows = rows_by_key(metadata_view(session))
        assert rows["sample_name"].status == "user" and rows["sample_name"].value == "granite"
        assert rows["voxel_size"].status in ("default", "missing")                 # the file had none
        assert rows["center_of_rotation"].status == "settings" and rows["center_of_rotation"].value == "12.5"
    finally:
        session.run.project.close()


def test_a_cone_beam_project_misses_its_distances(tmp_path):
    path = make_file(tmp_path / "c.h5", geometry_type="cone", angle_span=360.0, voxel_size=1.0)
    session = make_session(path)
    try:
        rows = rows_by_key(metadata_view(session))
        assert rows["source_to_detector_distance"].status == "missing"
        assert rows["source_to_object_distance"].status == "missing"
        assert "missing required fields" in metadata_view(session).summary          # the reason it cannot go on
    finally:
        session.run.project.close()
    session = make_session(path, source_to_detector_distance=300.0, source_to_object_distance=150.0)
    try:
        rows = rows_by_key(metadata_view(session))
        assert rows["source_to_detector_distance"].status == "user" and rows["source_to_detector_distance"].value == "300"
    finally:
        session.run.project.close()


def test_extra_metadata_is_listed_under_other(tmp_path):
    path = make_file(tmp_path / "d.h5", geometry_type="parallel")
    with h5py.File(path, "a") as f:
        f.create_dataset("/entry/extra/beam_current", data=250.0)
    session = make_session(path)
    try:
        rows = rows_by_key(metadata_view(session))
        assert rows["extra.beam_current"].group == "other" and rows["extra.beam_current"].value == "250"
    finally:
        session.run.project.close()


def test_widget_lists_groups_filters_and_follows_the_session(app, tmp_path):
    path = make_file(tmp_path / "e.h5", geometry_type="parallel", angles=np.linspace(0, 180, 6, endpoint=False), voxel_size=1.0)
    session = make_session(path)
    try:
        box = MetadataToolbox()
        assert "No project selected" in box.title_label.text() and box.tree.topLevelItemCount() == 0
        box.show_session(session)
        assert box.title_label.text() == session.project_id
        heads = [box.tree.topLevelItem(i).text(0) for i in range(box.tree.topLevelItemCount())]
        assert any(h.startswith("Must be in the file") for h in heads) and any("Optional" in h for h in heads)
        all_rows = sum(box.tree.topLevelItem(i).childCount() for i in range(box.tree.topLevelItemCount()))

        box.only_problems.setChecked(True)
        problem_rows = sum(box.tree.topLevelItem(i).childCount() for i in range(box.tree.topLevelItemCount()))
        assert 0 < problem_rows < all_rows
        box.only_problems.setChecked(False)

        session.config.completion_answers["operator"] = "me"             # a refresh of a different session is ignored
        box.refresh_if(ProjectSession(config=ProjectConfig(path="other.h5")))
        assert box.session is session
    finally:
        session.run.project.close()


def test_main_window_has_a_closable_dock_that_follows_the_selection(app, tmp_path):
    from PySide6.QtCore import Qt

    from gui.main_window import MainWindow

    path = make_file(tmp_path / "f.h5", geometry_type="parallel", angles=np.linspace(0, 180, 6, endpoint=False), voxel_size=1.0)
    session = make_session(path)
    window = MainWindow()
    try:
        assert window.metadata_dock.isHidden()
        entries = [a.text() for menu in window.menuBar().actions() if menu.menu() for a in menu.menu().actions()]
        assert "Metadata toolbox" in entries
        window.project_panel.add_session(session)
        window.project_panel.tree.topLevelItem(0).setSelected(True)
        assert window.metadata_toolbox.session is session
        assert window.metadata_toolbox.tree.topLevelItemCount() > 0
    finally:
        window.close()
        session.run.project.close()
