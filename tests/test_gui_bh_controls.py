"""Beam hardening controls in the Refinements tab and how the panel wires them."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
pytest.importorskip("algotom")
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.adjustment_panel import AdjustmentPanel, RefinementsTab  # noqa: E402
from gui.bh_info import summarize_beam_hardening  # noqa: E402
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
    config = ProjectConfig(path=str(path), format="hdf5", storage_mode="disk", center=16.0, tilt_deg=0.0)
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


# -- the summary ------------------------------------------------------------------------------------

def test_summary_off_on_and_unusable():
    off = summarize_beam_hardening(ProjectConfig())
    assert not off.ok and off.text.startswith("Off") and off.coefficients == (1.0, 0.0, 0.0)

    on = summarize_beam_hardening(ProjectConfig(beam_hardening=True, bh_coefficients=(1.0, 0.1)))
    assert on.ok and on.coefficients == (1.0, 0.1, 0.0)
    assert "1 -> 1.100" in on.text and "2 -> 2.400" in on.text       # p + 0.1 p^2

    bad = summarize_beam_hardening(ProjectConfig(beam_hardening=True, bh_coefficients=(1.0, -0.5)))
    assert not bad.ok and "Not usable" in bad.text


# -- the tab alone ----------------------------------------------------------------------------------

def test_controls_start_off_and_disabled(app):
    tab = RefinementsTab()
    assert not tab.bh_checkbox.isChecked() and all(not spin.isEnabled() for spin in tab.bh_spins)
    assert len(tab.bh_spins) == 3


def test_ticking_enables_the_fields_and_editing_emits_the_index(app):
    tab = RefinementsTab()
    toggled, edited = [], []
    tab.bh_toggled.connect(toggled.append)
    tab.bh_coefficient_changed.connect(lambda index, value: edited.append((index, value)))
    tab.bh_checkbox.setChecked(True)
    assert toggled == [True] and all(spin.isEnabled() for spin in tab.bh_spins)
    tab.bh_spins[1].setValue(0.25)
    tab.bh_spins[2].setValue(-0.01)
    assert edited == [(1, 0.25), (2, -0.01)]


def test_set_beam_hardening_shows_state_without_emitting(app):
    tab = RefinementsTab()
    seen = []
    tab.bh_toggled.connect(seen.append)
    tab.bh_coefficient_changed.connect(lambda *a: seen.append(a))
    tab.set_beam_hardening(True, (1.0, 0.3, -0.02), "some text", False)
    assert seen == [] and tab.bh_checkbox.isChecked()
    assert [round(spin.value(), 3) for spin in tab.bh_spins] == [1.0, 0.3, -0.02]
    assert tab.bh_info_label.text() == "some text" and "c0392b" in tab.bh_info_label.styleSheet()


# -- the panel --------------------------------------------------------------------------------------

def test_ticking_starts_from_the_identity_correction(panel, session):
    tab = panel.refinements_tab
    tab.bh_checkbox.setChecked(True)
    assert session.config.beam_hardening is True and session.config.bh_coefficients == (1.0, 0.0, 0.0)
    assert "Measured attenuation" in tab.bh_info_label.text()


def test_editing_coefficients_writes_a_tuple_into_the_config(panel, session):
    tab = panel.refinements_tab
    tab.bh_checkbox.setChecked(True)
    tab.bh_spins[1].setValue(0.3)
    tab.bh_spins[2].setValue(-0.02)
    assert session.config.bh_coefficients == (1.0, 0.3, -0.02)
    assert "1 -> 1.280" in tab.bh_info_label.text()                  # 1 + 0.3 - 0.02
    tab.bh_checkbox.setChecked(False)
    assert session.config.beam_hardening is False and session.config.bh_coefficients == (1.0, 0.3, -0.02)


def test_unusable_values_are_flagged_and_do_not_restart_the_preview(panel, session):
    tab = panel.refinements_tab
    tab.bh_checkbox.setChecked(True)
    panel._auto_update_timer.stop()
    panel._auto_update = True
    tab.bh_spins[1].setValue(0.3)
    assert panel._auto_update_timer.isActive()                       # usable: re-preview

    tab.bh_spins[0].setValue(0.05)
    panel._auto_update_timer.stop()
    tab.bh_spins[1].setValue(-2.0)                                   # stops increasing early
    assert "Not usable" in tab.bh_info_label.text() and "c0392b" in tab.bh_info_label.styleSheet()
    assert not panel._auto_update_timer.isActive()


def test_each_project_keeps_its_own_correction(app, panel, session):
    tab = panel.refinements_tab
    tab.bh_checkbox.setChecked(True)
    tab.bh_spins[1].setValue(0.4)

    other_config = ProjectConfig(path=session.config.path, format="hdf5", storage_mode="disk", center=16.0)
    other_run = PipelineRun(other_config)
    other_run.import_data()
    other_run.complete()
    other = ProjectSession(config=other_config, run=other_run)
    try:
        panel.set_session(other)
        assert not tab.bh_checkbox.isChecked() and round(tab.bh_spins[1].value(), 3) == 0.0
        panel.set_session(session)
        assert tab.bh_checkbox.isChecked() and round(tab.bh_spins[1].value(), 3) == 0.4
    finally:
        other_run.project.close()


def test_the_settings_can_be_copied_between_projects(session):
    assert [name for name, _ in COPYABLE_SETTINGS["Beam hardening"]] == ["beam_hardening", "bh_coefficients"]
    session.config.beam_hardening = True
    session.config.bh_coefficients = (1.0, 0.33, -0.02)
    clipboard = copy_settings(session, ["beam_hardening", "bh_coefficients"])
    target = ProjectSession(config=ProjectConfig(path="b.h5"))
    paste_settings(target, clipboard)
    assert target.config.beam_hardening is True and target.config.bh_coefficients == (1.0, 0.33, -0.02)
