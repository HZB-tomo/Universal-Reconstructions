"""The reusable imageviewer package: measuring, the histogram panel, the line tool and its profile window."""
import math
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from imageviewer import HistogramPanel, StackViewer  # noqa: E402
from imageviewer.histogram import compute_histogram, percentile_levels  # noqa: E402
from imageviewer.measure import describe_line, line_angle, line_length, line_profile  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def ramp_frame(h=40, w=60):
    """value = column index, so a profile along a row is easy to predict."""
    return np.tile(np.arange(w, dtype=np.float32), (h, 1))


# -- measuring, no Qt ----------------------------------------------------------

def test_length_and_angle_follow_the_screen():
    assert line_length((0, 0), (3, 4)) == 5.0
    assert line_angle((0, 0), (10, 0)) == 0.0
    assert line_angle((0, 0), (0, -10)) == pytest.approx(90.0)         # up on screen: rows run downward
    assert line_angle((0, 0), (0, 10)) == pytest.approx(-90.0)
    assert line_angle((0, 0), (-5, 0)) == pytest.approx(180.0)
    assert line_angle((0, 0), (5, -5)) == pytest.approx(45.0)


def test_describe_line_adds_the_physical_unit_when_known():
    assert describe_line((0, 0), (30, 40)) == "50.0 px, -53.1°"
    assert "= 100 um" in describe_line((0, 0), (30, 40), (2.0, "um"))


def test_profile_along_a_row_reads_pixel_centres():
    frame = ramp_frame()
    # a pixel covers x in [c, c+1]; its centre c + 0.5 holds value c
    distance, values = line_profile(frame, (10.5, 5.5), (20.5, 5.5))
    assert len(values) == 11 and distance[0] == 0 and distance[-1] == pytest.approx(10.0)
    np.testing.assert_allclose(values, np.arange(10, 21))
    _, between = line_profile(frame, (10.0, 5.5), (12.0, 5.5))
    np.testing.assert_allclose(between, [9.5, 10.5, 11.5])             # bilinear between centres


def test_profile_outside_the_frame_is_nan_and_width_averages_across():
    frame = ramp_frame(h=20, w=20)
    _, values = line_profile(frame, (15.5, 5.5), (25.5, 5.5))
    assert np.isfinite(values[:4]).all() and np.isnan(values[-1])
    vertical = np.tile(np.arange(20, dtype=np.float32)[:, None], (1, 20))     # value = row index
    _, thin = line_profile(vertical, (5.5, 5.5), (15.5, 5.5), width=1)
    _, wide = line_profile(vertical, (5.5, 5.5), (15.5, 5.5), width=5)
    np.testing.assert_allclose(thin, 5.0)
    np.testing.assert_allclose(wide, 5.0, atol=1e-6)                    # the mean of rows 3..7
    noisy = vertical + np.random.default_rng(0).normal(0, 1, vertical.shape)
    assert np.nanstd(line_profile(noisy, (2.5, 8.5), (17.5, 8.5), width=9)[1]) < np.nanstd(line_profile(noisy, (2.5, 8.5), (17.5, 8.5))[1])


# -- histogram numbers -------------------------------------------------------------

def test_histogram_counts_ignore_non_finite_values():
    data = np.array([0, 1, 1, 2, 2, 2, np.nan, np.inf], dtype=np.float32)
    h = compute_histogram(data, bins=3)
    assert h.n == 6 and h.minimum == 0 and h.maximum == 2 and h.counts.sum() == 6
    assert h.mean == pytest.approx(8 / 6)
    assert compute_histogram(np.full(5, np.nan)) is None and compute_histogram(None) is None


def test_percentile_levels_clip_the_ends():
    data = np.concatenate([np.linspace(0, 100, 10000), [1e6]])             # one hot pixel
    assert percentile_levels(data, 0)[1] == 1e6
    lo, hi = percentile_levels(data, 1.0)
    assert 0.5 < lo < 2 and 98 < hi < 100.5
    assert percentile_levels(np.full(4, 3.0)) == (3.0, 4.0)                  # flat image: a usable window


# -- canvas and histogram panel ------------------------------------------------------

def test_canvas_levels_lock_and_visible_area(app):
    viewer = StackViewer()
    viewer.resize(400, 300)
    viewer.show_image(ramp_frame())
    canvas = viewer.canvas
    assert canvas.levels == (0.0, 59.0)
    canvas.set_levels(10, 20)
    canvas.levels_locked = True
    viewer.show_image(ramp_frame() * 2)                                 # a new image, but the levels are locked
    assert canvas.levels == (10.0, 20.0)
    canvas.levels_locked = False
    viewer.show_image(ramp_frame() * 2)
    assert canvas.levels == (0.0, 118.0)
    canvas.view.setAspectLocked(False)             # a locked aspect widens the range to the window's shape
    canvas.view.setRange(xRange=(10, 20), yRange=(5, 15), padding=0)
    visible = canvas.visible_frame()
    assert visible.shape == (10, 10) and visible.min() == 20 and visible.max() == 38


def test_histogram_panel_buttons_and_controls(app):
    viewer = StackViewer()
    viewer.resize(500, 400)
    data = ramp_frame(h=40, w=100)
    data[0, 0] = 1000.0                                               # a hot pixel
    viewer.show_image(data)
    canvas, panel = viewer.canvas, viewer.make_histogram()
    assert isinstance(panel, HistogramPanel) and canvas.levels == (0.0, 1000.0)

    panel.scale_to_minmax()
    assert canvas.levels == (0.0, 1000.0) and panel.max_spin.value() == 1000.0
    panel.clip_spin.setValue(1.0)
    panel.auto_scale()
    assert canvas.levels[1] < 100 and panel.max_spin.value() == pytest.approx(canvas.levels[1])

    canvas.view.setAspectLocked(False)
    canvas.view.setRange(xRange=(0, 20), yRange=(0, 20), padding=0)
    panel.clip_spin.setValue(0.0)
    panel.scale_to_view()
    assert canvas.levels == (0.0, 1000.0)                                   # (0, 0) is in view
    canvas.view.setRange(xRange=(30, 50), yRange=(10, 20), padding=0)
    panel.scale_to_view()
    assert canvas.levels == (30.0, 49.0)

    panel.min_spin.setValue(5.0)                                            # typing the black point
    panel.min_spin.editingFinished.emit()
    assert canvas.levels[0] == 5.0
    panel.region.setRegion((7.0, 70.0))                                     # dragging the region
    assert canvas.levels == (7.0, 70.0) and panel.min_spin.value() == 7.0

    panel.lock_box.setChecked(True)
    assert canvas.levels_locked
    viewer.show_image(data * 2)
    assert canvas.levels == (7.0, 70.0)

    panel.map_box.setCurrentText("viridis")
    panel.invert_box.setChecked(True)
    assert canvas.colormap == ("viridis", True)


def test_histogram_describes_the_whole_image_or_the_visible_area(app):
    viewer = StackViewer()
    viewer.resize(500, 400)
    viewer.show_image(ramp_frame(h=40, w=100))
    panel = viewer.make_histogram()
    assert "4,000 pixels" in panel.stats_label.text() and panel._histogram.maximum == 99
    viewer.canvas.view.setAspectLocked(False)
    viewer.canvas.view.setRange(xRange=(0, 10), yRange=(0, 10), padding=0)
    panel.source_box.setCurrentIndex(1)
    panel.refresh()
    assert panel._histogram.maximum == 9 and "visible area" in panel.stats_label.text()
    panel.log_box.setChecked(False)
    assert panel.plot.getPlotItem().getAxis("left").labelText == "counts"


# -- line tool and profile ------------------------------------------------------------

def draw(viewer, p0, p1):
    tool = viewer.tools.tools["line"]
    tool.begin(p0)
    tool.move_to(p1)
    tool.finish()
    return viewer.tools.current


def test_line_tool_measures_length_and_angle(app):
    viewer = StackViewer()
    viewer.show_image(ramp_frame())
    line = draw(viewer, (10, 30), (40, 30))
    assert line.length == pytest.approx(30.0) and line.label.toPlainText() == "30.0 px, 0.0°"
    line = draw(viewer, (10, 30), (10, 10))                                # replaces the first
    assert viewer.tools.current is line and line.label.toPlainText() == "20.0 px, 90.0°"
    viewer.set_pixel_size(2.5, "um")                                        # the label picks up a physical size
    assert "= 50 um" in line.label.toPlainText()
    viewer.set_pixel_size(None)
    assert "um" not in line.label.toPlainText()
    viewer.tools.clear()
    assert viewer.tools.current is None


def test_a_click_is_not_a_line(app):
    viewer = StackViewer()
    viewer.show_image(ramp_frame())
    assert draw(viewer, (10, 10), (10.5, 10.5)) is None


def test_a_drag_with_the_tool_active_draws_through_the_viewbox(app):
    class Event:                                   # the bits of pyqtgraph's drag event the tool reads
        def __init__(self, button, start, pos, first=False, last=False):
            self._b, self._s, self._p, self._first, self._last = button, start, pos, first, last

        def button(self): return self._b
        def buttonDownPos(self): return self._s
        def pos(self): return self._p
        def isStart(self): return self._first
        def isFinish(self): return self._last
        def accept(self): self.accepted = True

    from PySide6.QtCore import QPointF

    viewer = StackViewer()
    viewer.resize(400, 300)
    viewer.show()
    viewer.show_image(ramp_frame())
    view = viewer.canvas.view
    assert view.active_tool is None
    viewer.tools.set_active("line")
    assert view.active_tool is viewer.tools.tools["line"]
    start = view.mapFromView(QPointF(10, 10))
    end = view.mapFromView(QPointF(30, 10))
    for ev in (Event(Qt.LeftButton, start, start, first=True), Event(Qt.LeftButton, start, end, last=True)):
        view.mouseDragEvent(ev)
        assert ev.accepted
    assert viewer.tools.current.length == pytest.approx(20.0, abs=0.5)
    viewer.tools.set_active(None)
    assert view.active_tool is None
    viewer.close()


def test_k_opens_a_live_profile_that_follows_the_line_and_the_slice(app):
    stack = np.stack([ramp_frame(), ramp_frame() + 100])
    viewer = StackViewer()
    viewer.resize(400, 300)
    viewer.show()
    viewer.show_stack(stack)
    line = draw(viewer, (10.5, 5.5), (20.5, 5.5))
    viewer.canvas.setFocus()
    QTest.keyClick(viewer.canvas, Qt.Key_K)
    window = viewer.tools._profile
    assert window is not None and window.isVisible()
    np.testing.assert_allclose(window.values, np.arange(10, 21))
    assert "10.0 px" in window.info_label.text()

    line.set_end((30.5, 5.5))                                               # move the end: the profile follows
    np.testing.assert_allclose(window.values, np.arange(10, 31))
    viewer._frame_slider.setValue(1)                                        # another slice: it follows too
    np.testing.assert_allclose(window.values, np.arange(10, 31) + 100)

    window.width_spin.setValue(3)
    assert window.values is not None and len(window.values) == 21
    viewer.set_pixel_size(2.0, "um")
    window.update_profile()
    assert "um" in window.info_label.text()

    viewer.tools.clear()
    assert window.values is None and "Draw a line" in window.info_label.text()
    QTest.keyClick(viewer.canvas, Qt.Key_K)                                 # K again reuses the same window
    assert viewer.tools._profile is window
    window.close()
    viewer.close()


def test_delete_and_escape_keys(app):
    viewer = StackViewer()
    viewer.show()
    viewer.show_image(ramp_frame())
    draw(viewer, (5, 5), (30, 20))
    viewer.tools.set_active("line")
    QTest.keyClick(viewer.canvas, Qt.Key_Escape)
    assert viewer.tools.active is None
    QTest.keyClick(viewer.canvas, Qt.Key_Delete)
    assert viewer.tools.current is None
    viewer.close()


def test_toolbar_buttons_follow_the_active_tool(app):
    viewer = StackViewer()
    bar = viewer.make_toolbar()
    actions = bar.actions_by_tool
    assert actions[None].isChecked() and not actions["line"].isChecked()
    actions["line"].trigger()
    assert viewer.tools.active.name == "line" and actions["line"].isChecked() and not actions[None].isChecked()
    actions[None].trigger()
    assert viewer.tools.active is None
    assert [a.text() for a in bar.actions() if a.text()] == ["Pan / zoom", "Line", "Profile (K)", "Clear"]


def test_physical_scale_is_only_used_where_a_pixel_is_a_length(app):
    stack = np.random.default_rng(0).random((6, 8, 10), dtype=np.float32)
    viewer = StackViewer()
    viewer.set_pixel_size(3.0)
    viewer.show_stack_lazy(lambda i: stack[i], 6, shape=stack.shape, kind="projections")
    assert viewer.canvas.physical_pixel() == (3.0, "um")
    viewer.set_axis(1)                                                      # a sinogram: angle x column
    assert viewer.canvas.physical_pixel() is None


# -- independence ----------------------------------------------------------------------

def test_the_package_needs_nothing_of_the_rest_of_the_repository():
    code = textwrap.dedent("""
        import sys
        import imageviewer, imageviewer.profile, imageviewer.measure
        bad = sorted(m for m in sys.modules if m.split('.')[0] in ('gui', 'core', 'orchestration', 'reconstruction', 'data_import'))
        print(bad)
    """)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    # make the repo importable the way an installed copy would be, then see what the package pulled in
    out = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, '.')\n" + code], capture_output=True, text=True,
                         env=env, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_main_window_docks_the_toolbar_and_the_histogram(app):
    from gui.main_window import MainWindow

    window = MainWindow()
    try:
        assert window.dockWidgetArea(window.histogram_dock) == Qt.BottomDockWidgetArea
        assert window.histogram_dock.isFloating() is False and window.histogram_dock.widget() is window.histogram_panel
        assert window.toolBarArea(window.tools_toolbar) == Qt.TopToolBarArea
        entries = [a.text() for menu in window.menuBar().actions() if menu.text() == "View" for a in menu.menu().actions()]
        assert "Histogram" in entries and "Image tools" in entries
        window.histogram_dock.setFloating(True)                                 # "can be separated"
        assert window.histogram_dock.isFloating()
    finally:
        window.close()
