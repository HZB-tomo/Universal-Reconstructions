"""Automatic-adjustments tab (center-of-rotation scan: metric, range, curve, slider) and the project panel's import shortcuts."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from calibration import CenterMetric, center_grid, list_metrics, register_metric, scan_centers  # noqa: E402
from gui.adjustment_panel import AutomaticAdjustmentsTab  # noqa: E402
from gui.center_tab import COLUMN, SHIFT  # noqa: E402
from gui.project_panel import ProjectPanel  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class _Peak(CenterMetric):
    """A made-up metric (lower is better) with its minimum at 70.2, to draw a curve without a project."""

    name = "gui_test_peak"
    label = "peak"
    kind = "sinogram"
    lower_is_better = True

    def curve(self, centers, **_):
        return (centers - 70.2) ** 2


register_metric(_Peak)


def _tab(width=128):
    tab = AutomaticAdjustmentsTab()
    tab.set_width(width)
    return tab


def _scan(lo=-10.0, hi=10.0, n=41, width=128):
    return scan_centers("gui_test_peak", center_grid(width, lo, hi, n))


def test_the_tab_lists_every_metric_with_its_cost_and_explains_it(app):
    tab = _tab()
    assert [tab.metric_combo.itemData(i) for i in range(tab.metric_combo.count())] == list_metrics()
    assert list_metrics()[0] == "vo"
    assert "[fast]" in tab.metric_combo.itemText(tab.metric_combo.findData("vo"))
    assert "[reconstructs each]" in tab.metric_combo.itemText(tab.metric_combo.findData("sharpness"))
    assert "Vo" in tab.metric_combo.itemData(tab.metric_combo.findData("vo"), Qt.ToolTipRole)


def test_the_default_range_is_an_eighth_of_the_width_either_side(app):
    tab = _tab(256)
    assert tab.default_range() == (-32.0, 32.0)
    tab.set_state("vo", tab.default_range(), 101, None, None)
    assert tab.shift_range() == (-32.0, 32.0) and tab.points() == 101 and tab.metric() == "vo"
    assert "101 trial centers" in tab.hint_label.text() and "0.64 px" in tab.hint_label.text()
    assert "coarser" in tab.hint_label.text()                                 # 0.64 px is more than 0.5
    tab.points_spin.setValue(201)
    assert "coarser" not in tab.hint_label.text() and "0.32 px" in tab.hint_label.text()


def test_slice_metrics_say_how_many_reconstructions_they_cost(app):
    tab = _tab()
    tab.metric_combo.setCurrentIndex(tab.metric_combo.findData("sharpness"))
    tab.points_spin.setValue(50)
    assert "50 reconstructions" in tab.hint_label.text()
    tab.metric_combo.setCurrentIndex(tab.metric_combo.findData("vo"))
    assert "reconstructions" not in tab.hint_label.text()


def test_scan_sends_the_metric_the_range_as_shifts_and_the_points(app):
    tab = _tab()
    seen = []
    tab.scan_requested.connect(lambda *args: seen.append(args))
    tab.metric_combo.setCurrentIndex(tab.metric_combo.findData("entropy"))
    tab.from_spin.setValue(-5.0)
    tab.to_spin.setValue(12.5)
    tab.points_spin.setValue(36)
    tab.scan_btn.click()
    assert seen == [("entropy", -5.0, 12.5, 36)]


def test_the_range_can_be_shown_as_detector_columns_but_is_always_sent_as_shifts(app):
    tab = _tab(128)
    tab.set_state("vo", (-5.0, 12.5), 36, None, None)
    tab.axis_combo.setCurrentText(COLUMN)
    assert (tab.from_spin.value(), tab.to_spin.value()) == (58.5, 76.0)        # the middle of 128 columns is 63.5
    assert tab.shift_range() == (-5.0, 12.5)
    tab.from_spin.setValue(60.5)                                               # typed as a column
    seen = []
    tab.scan_requested.connect(lambda *args: seen.append(args))
    tab.scan_btn.click()
    assert seen == [("vo", -3.0, 12.5, 36)]
    tab.axis_combo.setCurrentText(SHIFT)
    assert tab.shift_range() == (-3.0, 12.5) and tab.from_spin.value() == -3.0


def test_an_empty_range_is_refused(app):
    tab = _tab()
    seen = []
    tab.scan_requested.connect(lambda *args: seen.append(args))
    tab.from_spin.setValue(5.0)
    tab.to_spin.setValue(5.0)
    assert "empty" in tab.hint_label.text()
    tab.scan_btn.click()
    assert seen == [] and "empty" in tab.note_label.text()


def test_editing_the_settings_is_reported_but_loading_them_is_not(app):
    tab = _tab()
    changed = []
    tab.settings_changed.connect(lambda: changed.append(1))
    tab.set_state("pair_registration", (-3.0, 4.0), 55, None, None)
    assert changed == [] and tab.metric() == "pair_registration"
    tab.points_spin.setValue(60)
    tab.from_spin.setValue(-4.0)
    tab.metric_combo.setCurrentIndex(0)
    assert len(changed) == 3


def test_a_scan_draws_the_curve_and_puts_the_line_on_the_best_center(app):
    tab = _tab()
    scan = _scan()
    tab.set_scan(scan)
    x, y = tab.curve_item.getData()
    assert len(x) == 41 and x[0] == -10.0 and x[-1] == 10.0 and y.max() == 1.0       # x: shifts from the middle
    assert tab.current_line.value() == pytest.approx(scan.best - 63.5, abs=0.5)
    assert tab.slider.isEnabled() and tab.slider.maximum() == 40 and tab.zoom_btn.isEnabled() and tab.best_btn.isEnabled()
    assert "center of rotation 70" in tab.center_label.text() and "quality 1.0" in tab.center_label.text()
    tab.axis_combo.setCurrentText(COLUMN)
    x, _ = tab.curve_item.getData()
    assert x[0] == 53.5 and tab.current_line.value() == pytest.approx(scan.best, abs=0.5)


def test_the_slider_moves_the_center_and_reports_it_final_unless_dragged(app):
    tab = _tab()
    scan = _scan()
    tab.set_scan(scan)
    seen = []
    tab.center_scrubbed.connect(lambda c, final: seen.append((c, final)))
    tab.slider.setValue(0)
    assert seen[-1] == (scan.centers[0], True)               # a key press or a click: final
    assert tab.current_line.value() == pytest.approx(-10.0)
    tab.slider.setSliderDown(True)
    tab.slider.setValue(10)
    assert seen[-1] == (scan.centers[10], False)             # dragged: not final yet
    tab.slider.setSliderDown(False)
    tab.slider.sliderReleased.emit()
    assert seen[-1] == (scan.centers[10], True)
    assert "outside" not in tab.center_label.text()


def test_the_line_on_the_plot_is_draggable_and_clamped_to_the_scan(app):
    tab = _tab()
    scan = _scan()
    tab.set_scan(scan)
    seen = []
    tab.center_scrubbed.connect(lambda c, final: seen.append((c, final)))
    tab.current_line.setValue(2.0)                           # dragging
    assert seen[-1][0] == pytest.approx(63.5 + 2.0) and tab.slider.value() == int(np.argmin(np.abs(scan.centers - 65.5)))
    tab.current_line.sigPositionChangeFinished.emit(tab.current_line)
    assert seen[-1][1] is True


def test_following_a_center_changed_elsewhere_emits_nothing_and_flags_a_center_outside_the_scan(app):
    tab = _tab()
    tab.set_scan(_scan())
    seen = []
    tab.center_scrubbed.connect(lambda *a: seen.append(a))
    tab.set_current_center(60.0)
    assert seen == [] and tab.slider.value() == int(np.argmin(np.abs(tab.scan().centers - 60.0)))
    tab.set_current_center(100.0)
    assert "outside the scanned range" in tab.center_label.text()


def test_go_to_best_and_zoom_and_recentre(app):
    tab = _tab()
    scan = _scan()
    tab.set_state("gui_test_peak", (-10.0, 10.0), 41, scan, 60.0)
    seen, scans = [], []
    tab.center_scrubbed.connect(lambda c, final: seen.append((c, final)))
    tab.scan_requested.connect(lambda *args: scans.append(args))
    tab.best_btn.click()
    assert seen == [(scan.best, True)]
    tab.zoom_btn.click()                                     # a tenth as wide, around the best
    metric, lo, hi, n = scans[-1]
    assert metric == "gui_test_peak" and n == 41 and hi - lo == pytest.approx(2.0)
    assert (lo + hi) / 2 == pytest.approx(scan.best - 63.5, abs=1e-6)
    tab.set_current_center(80.0)
    tab.recentre_btn.click()                                 # keeps the width, moves it onto the current center
    assert tab.shift_range() == pytest.approx((80.0 - 63.5 - 1.0, 80.0 - 63.5 + 1.0))


def test_busy_turns_the_scan_button_into_cancel_and_shows_progress(app):
    tab = _tab()
    cancels, scans = [], []
    tab.scan_cancel_requested.connect(lambda: cancels.append(1))
    tab.scan_requested.connect(lambda *a: scans.append(a))
    tab.set_busy(True)
    assert tab.scan_btn.text() == "Cancel scan" and not tab.progress.isHidden() and not tab.from_spin.isEnabled()
    tab.set_progress(3, 40)
    assert tab.progress.value() == 3 and tab.progress.maximum() == 40 and "3 / 40" in tab.progress.format()
    tab.scan_btn.click()
    assert cancels == [1] and scans == []
    tab.set_busy(False)
    assert tab.scan_btn.text() == "Scan centers" and tab.progress.isHidden() and tab.from_spin.isEnabled()


def test_nothing_works_before_a_project_is_open(app):
    tab = AutomaticAdjustmentsTab()
    assert not tab.scan_btn.isEnabled() and not tab.slider.isEnabled() and not tab.zoom_btn.isEnabled()
    tab.set_scan(None)


def test_find_tilt_button_is_still_there(app):
    tab = _tab()
    seen = []
    tab.find_tilt_requested.connect(lambda: seen.append(1))
    tab.find_tilt_btn.click()
    assert seen == [1]


def test_double_click_on_the_empty_project_area_requests_a_disk_import(app):
    panel = ProjectPanel()
    panel.resize(300, 400)
    panel.show()
    requests = []
    panel.import_requested.connect(requests.append)
    QTest.mouseDClick(panel.tree.viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(20, 100))
    assert requests == ["disk"]


def test_right_click_menu_on_the_empty_area_offers_disk_and_ram(app):
    panel = ProjectPanel()
    requests = []
    panel.import_requested.connect(requests.append)
    actions = panel._build_import_menu().actions()
    assert [a.text() for a in actions] == ["Import project...", "Import project to RAM..."]
    for action in actions:
        action.trigger()
    assert requests == ["disk", "ram"]


def test_loading_a_default_range_is_not_an_edit(app):
    tab = AutomaticAdjustmentsTab()
    changed = []
    tab.settings_changed.connect(lambda: changed.append(1))
    tab.set_width(200)
    assert tab.shift_range() == (-25.0, 25.0) and changed == []


def test_auto_find_center_is_one_click_and_waits_for_a_project(app):
    tab = AutomaticAdjustmentsTab()
    assert not tab.auto_btn.isEnabled()
    tab.set_width(128)
    seen = []
    tab.auto_requested.connect(lambda: seen.append(1))
    tab.auto_btn.click()
    assert seen == [1]
    tab.set_busy(True)
    assert not tab.auto_btn.isEnabled()
    tab.set_busy(False)
    assert tab.auto_btn.isEnabled()


def test_the_quick_estimate_uses_the_range_only_once_it_has_been_typed(app):
    tab = _tab()
    assert tab.auto_window() is None                         # the default window: a tenth of the width either side
    tab.set_state("vo", (-5.0, 5.0), 40, None, None)         # loaded from a project: not typing
    assert tab.auto_window() is None
    tab.to_spin.setValue(9.0)
    assert tab.auto_window() == (-5.0, 9.0)
    tab.points_spin.setValue(30)                             # points do not matter to the window
    tab.set_state("vo", (-5.0, 5.0), 40, None, None)
    assert tab.auto_window() is None
    tab.set_state("vo", (-5.0, 5.0), 40, None, None, range_edited=True)         # a project that had one typed
    assert tab.auto_window() == (-5.0, 5.0)
    tab.axis_combo.setCurrentText(COLUMN)                    # showing it in columns is not an edit and changes no shifts
    assert tab.auto_window() == (-5.0, 5.0)
    tab.set_state("vo", (-5.0, 5.0), 40, None, None)
    tab.set_scan(_scan())
    tab.zoom_btn.click()                                     # a zoom sets the range, but the person did not type it
    assert not tab.range_edited
