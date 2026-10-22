"""ImageViewer: zoom kept across same-shape previews; slicing direction,
slice number and zoom saved/restored with view_state()/restore_view()."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.viewer import ImageViewer  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def viewer(app):
    v = ImageViewer()
    v.resize(600, 500)
    v.show()
    app.processEvents()
    return v


def _ranges(v):
    return [tuple(round(float(x), 3) for x in r) for r in v._view.view.viewRange()]


def test_zoom_survives_a_preview_of_the_same_shape(viewer, app):
    viewer.show_image(np.random.rand(40, 40).astype("float32"))
    viewer._view.view.setRange(xRange=(5, 20), yRange=(5, 20), padding=0)
    app.processEvents()
    zoomed = _ranges(viewer)
    viewer.show_image(np.random.rand(40, 40).astype("float32"))
    app.processEvents()
    assert _ranges(viewer) == zoomed


def test_a_preview_of_a_different_shape_resets_the_view(viewer, app):
    viewer.show_image(np.random.rand(40, 40).astype("float32"))
    viewer._view.view.setRange(xRange=(5, 20), yRange=(5, 20), padding=0)
    zoomed = _ranges(viewer)
    viewer.show_image(np.random.rand(80, 90).astype("float32"))
    app.processEvents()
    assert _ranges(viewer) != zoomed


def test_view_state_roundtrip_restores_axis_slice_and_zoom(viewer, app):
    volume = np.random.rand(10, 30, 40).astype("float32")
    viewer.show_stack(volume)
    viewer._axis_buttons[1].click()
    viewer._frame_slider.setValue(12)
    viewer._view.view.setRange(xRange=(2, 30), yRange=(1, 8), padding=0)
    app.processEvents()
    state = viewer.view_state()
    assert state["axis"] == 1 and state["index"] == 12

    viewer.show_stack(np.random.rand(10, 30, 40).astype("float32"))  # another project: back to z
    assert viewer.axis() == 0
    viewer.restore_view(state)
    app.processEvents()
    assert viewer.axis() == 1 and viewer._frame_slider.value() == 12
    assert [bool(b.isChecked()) for b in viewer._axis_buttons] == [False, True, False]
    assert _ranges(viewer)[0] == pytest.approx((2.0, 30.0), abs=0.01)


def test_restore_clamps_the_slice_to_the_new_stack(viewer):
    viewer.show_stack(np.random.rand(10, 30, 40).astype("float32"))
    viewer._axis_buttons[2].click()
    viewer._frame_slider.setValue(35)
    state = viewer.view_state()
    viewer.show_stack(np.random.rand(10, 30, 20).astype("float32"))  # narrower: only 20 x sections
    viewer.restore_view(state)
    assert viewer.axis() == 2 and viewer._frame_slider.value() == 19


def test_a_preview_image_has_no_state_to_save(viewer):
    viewer.show_image(np.random.rand(20, 20).astype("float32"))
    assert viewer.view_state() is None
    viewer.restore_view({"axis": 1, "index": 3, "frame_shape": (20, 20), "x_range": (0, 1), "y_range": (0, 1)})
    assert viewer.axis() == 0
