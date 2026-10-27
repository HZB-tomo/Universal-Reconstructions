"""Phase retrieval controls in the Refinements tab and how the panel wires them."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
pytest.importorskip("algotom")
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.adjustment_panel import AdjustmentPanel, RefinementsTab  # noqa: E402
from gui.phase_info import summarize_phase  # noqa: E402
from gui.session import ProjectSession  # noqa: E402
from gui.settings_clipboard import COPYABLE_SETTINGS, copy_settings, paste_settings  # noqa: E402
from gui.viewer import ImageViewer  # noqa: E402
from orchestration import PipelineRun, ProjectConfig  # noqa: E402
from tests.fixtures.realistic_phantom import generate as generate_phantom  # noqa: E402
from tests.fixtures.realistic_phantom import write_hdf5  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def session(tmp_path):
    path = tmp_path / "scan.h5"
    write_hdf5(generate_phantom(
        shape=(16, 32, 32), n_angles=40, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=1,
    ), path)
    config = ProjectConfig(
        path=str(path), format="hdf5", storage_mode="disk", center=16.0, tilt_deg=0.0,
        completion_answers={"energy": "25", "propagation_distance": "120"},
    )
    run = PipelineRun(config)
    run.import_data()
    run.complete()
    yield ProjectSession(config=config, run=run)
    run.project.close()


@pytest.fixture
def panel(app, session):
    p = AdjustmentPanel(ImageViewer())
    p.set_session(session)
    return p


# -- the tab alone ---------------------------------------------------------------------------

def test_phase_controls_start_off_and_disabled(app):
    tab = RefinementsTab()
    assert not tab.phase_checkbox.isChecked()
    assert not tab.phase_method_combo.isEnabled()
    assert all(not spin.isEnabled() for spin in tab.phase_spins.values())
    assert [tab.phase_method_combo.itemText(i) for i in range(tab.phase_method_combo.count())] == ["paganin"]


def test_ticking_the_box_enables_the_fields_and_emits(app):
    tab = RefinementsTab()
    seen = []
    tab.phase_toggled.connect(seen.append)
    tab.phase_checkbox.setChecked(True)
    assert seen == [True]
    assert tab.phase_method_combo.isEnabled() and all(spin.isEnabled() for spin in tab.phase_spins.values())


def test_editing_a_value_emits_its_key_and_zero_reads_as_not_set(app):
    tab = RefinementsTab()
    seen = []
    tab.phase_value_changed.connect(lambda key, value: seen.append((key, value)))
    tab.phase_spins["delta_beta"].setValue(150.0)
    tab.phase_spins["energy"].setValue(25.5)
    assert seen == [("delta_beta", 150.0), ("energy", 25.5)]
    assert tab.phase_spins["distance"].value() == 0.0 and tab.phase_spins["distance"].text() == "not set"


def test_set_phase_shows_state_without_emitting(app):
    tab = RefinementsTab()
    seen = []
    tab.phase_toggled.connect(seen.append)
    tab.phase_value_changed.connect(lambda *a: seen.append(a))
    tab.set_phase(True, "paganin", {"delta_beta": 200.0, "distance": 120.0, "energy": None, "pixel": 1.1},
                  {"delta_beta": "set here", "distance": "from the file", "energy": "", "pixel": "set here"},
                  "Still needed: energy.", False)
    assert seen == []
    assert tab.phase_checkbox.isChecked() and tab.phase_spins["delta_beta"].value() == 200.0
    assert tab.phase_spins["energy"].value() == 0.0
    assert tab.phase_origin_labels["distance"].text() == "(from the file)"
    assert tab.phase_origin_labels["energy"].text() == ""
    assert tab.phase_info_label.text() == "Still needed: energy."
    assert "c0392b" in tab.phase_info_label.styleSheet()        # incomplete settings are shown in red


# -- the summary shown under the controls ---------------------------------------------------------

def test_summary_says_what_is_missing_and_then_what_it_will_do(session):
    project = session.run.project
    off = summarize_phase(project, session.config)
    assert not off.ok and off.text.startswith("Off")

    session.config.phase_retrieval = True
    session.config.phase_pixel_um = 1.1
    needs = summarize_phase(project, session.config)
    assert not needs.ok and needs.missing == ["delta_over_beta"] and "delta/beta" in needs.text
    assert needs.values["energy_kev"] == (25.0, "user") and needs.values["distance_mm"] == (120.0, "user")
    assert needs.values["pixel_um"] == (1.1, "config")

    session.config.phase_delta_beta = 200.0
    ready = summarize_phase(project, session.config)
    assert ready.ok and "Smoothing length" in ready.text and "extra rows" in ready.text


# -- the panel ------------------------------------------------------------------------------------

def test_values_known_from_the_project_prefill_the_controls(panel):
    tab = panel.refinements_tab
    assert tab.phase_spins["energy"].value() == 25.0
    assert tab.phase_origin_labels["energy"].text() == "(typed in)"
    assert tab.phase_spins["distance"].value() == 120.0
    assert tab.phase_spins["pixel"].value() > 0 and tab.phase_origin_labels["pixel"].text() == "(from the file)"
    assert tab.phase_spins["delta_beta"].value() == 0.0         # no default exists for this one


def test_toggling_and_editing_write_into_the_config(panel, session):
    tab = panel.refinements_tab
    messages = []
    panel.status_message.connect(messages.append)
    tab.phase_checkbox.setChecked(True)
    assert session.config.phase_retrieval is True
    assert any("delta/beta" in m for m in messages)             # tells the user what is still needed
    assert "Still needed" in tab.phase_info_label.text()

    tab.phase_spins["delta_beta"].setValue(180.0)
    assert session.config.phase_delta_beta == 180.0
    assert "Smoothing length" in tab.phase_info_label.text()    # now complete: the summary changes

    tab.phase_spins["energy"].setValue(30.0)
    assert session.config.phase_energy_kev == 30.0
    assert tab.phase_origin_labels["energy"].text() == "(set here)"

    tab.phase_checkbox.setChecked(False)
    assert session.config.phase_retrieval is False


def test_setting_a_value_to_zero_goes_back_to_the_files_value(panel, session):
    tab = panel.refinements_tab
    tab.phase_spins["energy"].setValue(30.0)
    assert session.config.phase_energy_kev == 30.0
    tab.phase_spins["energy"].setValue(0.0)
    assert session.config.phase_energy_kev is None
    assert tab.phase_spins["energy"].value() == 25.0            # the value the project itself provides
    assert tab.phase_origin_labels["energy"].text() == "(typed in)"


def test_changing_settings_restarts_the_preview_when_auto_update_is_on(panel, session):
    tab = panel.refinements_tab
    tab.phase_checkbox.setChecked(True)
    tab.phase_spins["delta_beta"].setValue(100.0)               # complete now
    panel._auto_update_timer.stop()
    panel._auto_update = True
    tab.phase_spins["delta_beta"].setValue(150.0)
    assert panel._auto_update_timer.isActive()

    panel._auto_update_timer.stop()
    tab.phase_spins["delta_beta"].setValue(0.0)                 # incomplete: no point previewing
    assert not panel._auto_update_timer.isActive()


def test_other_session_shows_its_own_phase_settings(app, panel, session, tmp_path):
    tab = panel.refinements_tab
    tab.phase_checkbox.setChecked(True)
    tab.phase_spins["delta_beta"].setValue(150.0)

    other_config = ProjectConfig(path=session.config.path, format="hdf5", storage_mode="disk", center=16.0)
    other_run = PipelineRun(other_config)
    other_run.import_data()
    other_run.complete()
    other = ProjectSession(config=other_config, run=other_run)
    try:
        panel.set_session(other)
        assert not tab.phase_checkbox.isChecked() and tab.phase_spins["delta_beta"].value() == 0.0
        panel.set_session(session)
        assert tab.phase_checkbox.isChecked() and tab.phase_spins["delta_beta"].value() == 150.0
    finally:
        other_run.project.close()


# -- copy / paste of settings ---------------------------------------------------------------------

def test_phase_settings_can_be_copied_between_projects(session):
    assert [name for name, _ in COPYABLE_SETTINGS["Phase retrieval"]] == [
        "phase_retrieval", "phase_method", "phase_delta_beta", "phase_distance_mm", "phase_energy_kev", "phase_pixel_um",
    ]
    session.config.phase_retrieval = True
    session.config.phase_delta_beta = 175.0
    clipboard = copy_settings(session, [name for name, _ in COPYABLE_SETTINGS["Phase retrieval"]])
    target = ProjectSession(config=ProjectConfig(path="b.h5"))
    paste_settings(target, clipboard)
    assert target.config.phase_retrieval is True and target.config.phase_delta_beta == 175.0
