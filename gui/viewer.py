"""
ImageViewer: the viewer as this application uses it -- imageviewer.StackViewer (slider, z / y / x
slicing, cursor readout, line tool, histogram, all in the reusable imageviewer package) plus the one
thing specific to reconstruction: the crop rectangle drawn on the xy view.

Everything else about viewing belongs in imageviewer/, so that it can be changed there without
touching this application, and used in other projects.
"""
from __future__ import annotations

from typing import Optional

import pyqtgraph as pg
from PySide6.QtCore import Signal

from imageviewer import StackViewer


class ImageViewer(StackViewer):
    # Fires whenever the crop box is dragged/resized while visible, and
    # once right when it's shown (so a caller doesn't have to also call
    # get_crop_rect() itself just to seed an initial value). Not emitted
    # while hidden -- see set_crop_visible().
    crop_rect_changed = Signal(object)  # (y0, y1, x0, x1), half-open, or None

    def __init__(self, parent=None):
        super().__init__(parent)
        # The crop is an xy rectangle, so the box is only drawn in the z view (axis 0);
        # _crop_enabled is whether the user has it turned on, regardless of view.
        self._crop_roi = pg.RectROI([0, 0], [1, 1], pen=pg.mkPen("y", width=2))
        self._crop_roi.setVisible(False)
        self._crop_enabled = False
        self.canvas.view.addItem(self._crop_roi)
        self._crop_roi.sigRegionChangeFinished.connect(self._on_crop_roi_changed)

    def _axis_changed(self) -> None:
        self._apply_crop_visibility()

    # -- crop rectangle -------------------------------------------------------

    def _apply_crop_visibility(self) -> None:
        self._crop_roi.setVisible(self._crop_enabled and self._axis == 0)

    def set_crop_visible(self, visible: bool) -> None:
        """
        Show/hide the crop box with no particular rect in mind -- used
        the first time a session turns cropping on and has no stored
        rect yet. Seeds a non-degenerate default (80% of the current
        frame, centered) rather than a zero-size box. Emits
        crop_rect_changed with the result, same as a user drag would,
        so a caller doesn't also need to call get_crop_rect() itself.
        Switches back to the z view first: the crop is an xy rectangle.
        """
        if visible and self._axis != 0:
            self.set_axis(0)
        if visible and self.canvas.frame is not None:
            height, width = self.canvas.frame.shape[-2], self.canvas.frame.shape[-1]
            margin_y, margin_x = height * 0.1, width * 0.1
            self._crop_roi.setPos([margin_x, margin_y])
            self._crop_roi.setSize([width - 2 * margin_x, height - 2 * margin_y])
        self._crop_enabled = visible
        self._apply_crop_visibility()
        self.crop_rect_changed.emit(self.get_crop_rect() if visible else None)

    def set_crop_rect(self, rect: Optional[tuple[int, int, int, int]]) -> None:
        """
        Show the crop box at an exact (y0, y1, x0, x1), or hide it if
        `rect` is None -- for restoring a session's already-stored
        rect (ProjectSession.export_crop_yx) when switching to it.
        Does NOT emit crop_rect_changed: this is a sync FROM already-
        known state, not a new user choice, the same distinction
        BasicSettingsTab.set_values() draws for its own fields via
        blockSignals.
        """
        self._crop_roi.blockSignals(True)
        if rect is None:
            self._crop_enabled = False
        else:
            y0, y1, x0, x1 = rect
            self._crop_roi.setPos([x0, y0])
            self._crop_roi.setSize([x1 - x0, y1 - y0])
            self._crop_enabled = True
        self._apply_crop_visibility()
        self._crop_roi.blockSignals(False)

    def get_crop_rect(self) -> Optional[tuple[int, int, int, int]]:
        """
        (y0, y1, x0, x1), half-open -- or None if cropping is off.
        Still returned while the y / x view is showing (the box is just
        not drawn there). Deliberately NOT clipped to the currently displayed
        frame's own shape: which frame that is (a raw, lazily-loaded
        projection, or a reconstructed preview) changes independently
        of the crop box's own position/size every time a project is
        opened or reopened, confirmed directly to cause spurious
        empty/None results if this clamped against it -- a rect meant
        for a 300x300 reconstructed preview would get silently zeroed
        out the moment the viewer reverted to showing a (32, 300) raw
        projection stack instead. An out-of-range crop is harmless
        downstream regardless: numpy slicing clips silently, never
        raises, wherever config.crop is actually applied
        (orchestration/pipeline.py's postprocess()).
        """
        if not self._crop_enabled:
            return None
        pos, size = self._crop_roi.pos(), self._crop_roi.size()
        y0, x0 = max(0, int(round(pos.y()))), max(0, int(round(pos.x())))
        y1, x1 = int(round(pos.y() + size.y())), int(round(pos.x() + size.x()))
        if y1 <= y0 or x1 <= x0:
            return None
        return (y0, y1, x0, x1)

    def _on_crop_roi_changed(self) -> None:
        if self._crop_enabled and self._axis == 0:
            self.crop_rect_changed.emit(self.get_crop_rect())
