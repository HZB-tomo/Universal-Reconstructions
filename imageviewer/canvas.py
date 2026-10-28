"""
ImageCanvas: shows one 2D frame with pan/zoom, contrast levels and a colormap, and
tells whoever is interested what happens to it. It knows nothing about stacks,
sliders or histograms; those are separate pieces that talk to it through its
properties and signals:

    canvas.frame                 the 2D array on screen
    canvas.levels / set_levels   the contrast window (black point, white point)
    canvas.levels_locked         keep the levels when a new frame arrives
    canvas.visible_frame()       the part of the frame inside the current zoom
    canvas.view                  the ViewBox: add your own graphics items here
    frame_changed, levels_changed, view_changed, hovered, key_pressed

Part of the reusable imageviewer package: depends only on numpy, pyqtgraph and Qt.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QLineEdit


class ToolViewBox(pg.ViewBox):
    """A ViewBox whose left-button drags can be taken over by a tool (see tools.py);
    with no tool active it pans like any ViewBox."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_tool = None

    def mouseDragEvent(self, ev, axis=None):
        tool = self.active_tool
        if tool is not None and ev.button() == Qt.LeftButton:
            ev.accept()
            tool.drag(ev, self)
        else:
            super().mouseDragEvent(ev, axis)


def finite_range(frame: np.ndarray) -> tuple:
    """(min, max) of the finite values of `frame`; (0, 1) if there are none."""
    finite = frame[np.isfinite(frame)] if frame.dtype.kind == "f" else frame
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = float(finite.min()), float(finite.max())
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def make_colormap(name: str = "gray", inverted: bool = False) -> pg.ColorMap:
    if name == "gray":
        cmap = pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 255, 255)])
    else:
        cmap = pg.colormap.get(name)
    if inverted:
        cmap.reverse()
    return cmap


class ImageCanvas(pg.GraphicsView):
    frame_changed = Signal()
    levels_changed = Signal(float, float)
    view_changed = Signal()
    hovered = Signal(object)       # (column, row, value), or None when the cursor is off the image
    key_pressed = Signal(int)      # a Qt.Key_* pressed while the canvas has the keyboard focus

    COLORMAPS = ("gray", "viridis", "inferno", "magma", "plasma", "cividis")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = ToolViewBox()
        self.view.setAspectLocked(True)
        self.view.invertY(True)                    # row 0 at the top, like the array
        self.setCentralItem(self.view)
        self.image_item = pg.ImageItem(axisOrder="row-major")      # x = column, y = row
        self.view.addItem(self.image_item)

        self._frame: Optional[np.ndarray] = None
        self._levels: Optional[tuple] = None
        self.levels_locked = False
        # Size of one pixel in a physical unit, for measurements; None = unknown. The stack viewer
        # clears `scale_valid` for sections where a pixel is not a length (e.g. an angle axis).
        self.pixel_size: Optional[float] = None
        self.pixel_unit = "um"
        self.scale_valid = True
        self._colormap_name, self._inverted = "gray", False
        self.image_item.setColorMap(make_colormap())

        self.setFocusPolicy(Qt.StrongFocus)
        self.view.sigRangeChanged.connect(lambda *_: self.view_changed.emit())
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)

    # -- the frame ----------------------------------------------------------

    @property
    def frame(self) -> Optional[np.ndarray]:
        return self._frame

    def set_frame(self, frame: np.ndarray, reset_levels: bool = False, reset_view: bool = False, keep_zoom: bool = False) -> None:
        """Show `frame`. `reset_levels`: pick new levels from the data (unless locked);
        `reset_view`: zoom to fit (unless `keep_zoom`)."""
        self._frame = np.asarray(frame)
        changed = False
        if self._levels is None or (reset_levels and not self.levels_locked):
            self._levels, changed = finite_range(self._frame), True
        self.image_item.setImage(self._frame, autoLevels=False, levels=self._levels)
        if reset_view and not keep_zoom:
            self.view.autoRange(padding=0.02)
        self.frame_changed.emit()
        if changed:
            self.levels_changed.emit(*self._levels)

    def clear(self) -> None:
        self._frame = None
        self.image_item.clear()
        self.frame_changed.emit()

    # -- contrast -------------------------------------------------------------

    @property
    def levels(self) -> Optional[tuple]:
        return self._levels

    def set_levels(self, lo: float, hi: float) -> None:
        if not hi > lo:
            return
        self._levels = (float(lo), float(hi))
        self.image_item.setLevels(self._levels)
        self.levels_changed.emit(*self._levels)

    def set_colormap(self, name: str = "gray", inverted: bool = False) -> None:
        self._colormap_name, self._inverted = name, inverted
        self.image_item.setColorMap(make_colormap(name, inverted))

    @property
    def colormap(self) -> tuple:
        return self._colormap_name, self._inverted

    # -- the zoom ----------------------------------------------------------------

    def visible_slices(self) -> Optional[tuple]:
        """(row slice, column slice) of the frame inside the current zoom, or None when empty."""
        if self._frame is None:
            return None
        (x0, x1), (y0, y1) = self.view.viewRange()
        height, width = self._frame.shape[:2]
        c0, c1 = max(0, int(np.floor(x0))), min(width, int(np.ceil(x1)))
        r0, r1 = max(0, int(np.floor(y0))), min(height, int(np.ceil(y1)))
        if c1 <= c0 or r1 <= r0:
            return None
        return slice(r0, r1), slice(c0, c1)

    def visible_frame(self) -> Optional[np.ndarray]:
        """The part of the frame inside the current zoom (a view, not a copy)."""
        part = self.visible_slices()
        return None if part is None else self._frame[part]

    def physical_pixel(self) -> Optional[tuple]:
        """(size, unit) of one pixel, or None when unknown or meaningless for this section."""
        if self.pixel_size and self.scale_valid:
            return float(self.pixel_size), self.pixel_unit
        return None

    # -- mouse and keyboard -----------------------------------------------------------

    def _on_mouse_moved(self, scene_pos) -> None:
        if self._frame is None:
            self.hovered.emit(None)
            return
        point = self.view.mapSceneToView(scene_pos)
        col, row = int(np.floor(point.x())), int(np.floor(point.y()))
        height, width = self._frame.shape[:2]
        if 0 <= row < height and 0 <= col < width:
            self.hovered.emit((col, row, self._frame[row, col]))
        else:
            self.hovered.emit(None)

    def enterEvent(self, ev) -> None:
        # So that a key such as K works as soon as the cursor is over the image, unless the user is typing.
        focus = QApplication.focusWidget()
        if not isinstance(focus, (QLineEdit, QAbstractSpinBox)):
            self.setFocus(Qt.MouseFocusReason)
        super().enterEvent(ev)

    def keyPressEvent(self, ev) -> None:
        self.key_pressed.emit(ev.key())
        super().keyPressEvent(ev)
