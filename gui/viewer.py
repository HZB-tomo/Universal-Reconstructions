"""
ImageViewer: shows raw projections, a sinogram, or a reconstructed
slice/stack. A thin wrapper over pyqtgraph's ImageView, which already
provides the histogram/levels control and ROI tools. Its own frame
slider is hidden and replaced by ours (below): a single slider/display
path is used for both an eager, fully-in-memory stack (a reconstructed
volume -- reconstruct() always returns the whole thing, nothing lazy
to preserve there) and a lazy one (raw projections; see
show_stack_lazy()), rather than juggling two.

The z / y / x buttons choose which way a 3D stack is sliced (see
gui/slicing.py); z -- the stack's first axis -- is the default.
"""
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from gui.slicing import AXIS_NAMES, axis_labels, axis_tooltip, section_index

# A numpy array, or a callable taking a numpy-style index (an int, or a
# tuple such as (slice(None), j)) and returning that section.
FrameSource = Union[np.ndarray, Callable[[object], np.ndarray]]


class ImageViewer(QWidget):
    # Fires whenever the crop box is dragged/resized while visible, and
    # once right when it's shown (so a caller doesn't have to also call
    # get_crop_rect() itself just to seed an initial value). Not emitted
    # while hidden -- see set_crop_roi_visible().
    crop_rect_changed = Signal(object)  # (y0, y1, x0, x1), half-open, or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._view = pg.ImageView()
        self._view.ui.roiPlot.hide()  # replaced by our own slider below
        self._view.ui.roiBtn.hide()  # pg's own ROI button/plot -- unused, we have our own crop_roi below
        self._view.ui.menuBtn.hide()  # pg's own export/context menu button -- not needed

        # Our own crop rectangle -- deliberately separate from
        # pg.ImageView's built-in .roi, which is wired to the roiPlot
        # we hide above and isn't reusable for this. Sized/positioned
        # against whatever frame is showing when first made visible;
        # see set_crop_roi_visible(). The crop is an xy rectangle, so
        # the box is only drawn in the z view (axis 0); _crop_enabled
        # is whether the user has it turned on, regardless of view.
        self._crop_roi = pg.RectROI([0, 0], [1, 1], pen=pg.mkPen("y", width=2))
        self._crop_roi.setVisible(False)
        self._crop_enabled = False
        self._view.view.addItem(self._crop_roi)
        self._crop_roi.sigRegionChangeFinished.connect(self._on_crop_roi_changed)

        self._source: Optional[FrameSource] = None
        self._shape: Optional[tuple] = None  # (n0, n1, n2) of a 3D source; None = not sliceable
        self._kind = "volume"  # or "projections"; only changes the buttons' tooltips
        self._axis = 0
        self._single = False  # showing one preview image (show_image), not a browsable stack
        self._aspect_locked = True
        self._n_frames = 0
        self._current_frame: Optional[np.ndarray] = None  # the displayed frame, cached for the hover readout

        # Look like ordinary push buttons but behave as radio buttons:
        # exactly one stays pushed in, the others pop out.
        self._axis_buttons = []
        self._axis_group = QButtonGroup(self)
        self._axis_group.setExclusive(True)
        axis_row = QHBoxLayout()
        for axis, name in enumerate(AXIS_NAMES):
            button = QPushButton(name, self)
            button.setCheckable(True)
            button.setMaximumWidth(40)
            self._axis_group.addButton(button, axis)
            self._axis_buttons.append(button)
            axis_row.addWidget(button)
        axis_row.addStretch()
        self._axis_buttons[0].setChecked(True)
        self._axis_group.idClicked.connect(self._on_axis_clicked)

        self._frame_label = QLabel("frame: -", self)
        self._frame_slider = QSlider(Qt.Horizontal, self)
        self._frame_slider.valueChanged.connect(lambda i: self._show_frame(i, reset_view=False))

        self._readout = QLabel("x: -, y: -, value: -", self)

        frame_row = QHBoxLayout()
        frame_row.addWidget(self._frame_label)
        frame_row.addWidget(self._frame_slider)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(axis_row)
        layout.addWidget(self._view)
        layout.addLayout(frame_row)
        layout.addWidget(self._readout)

        self._update_axis_buttons()
        self._view.scene.sigMouseMoved.connect(self._on_mouse_moved)

    def show_stack(self, array: np.ndarray) -> None:
        """A 3D array already fully in memory -- e.g. a reconstructed
        volume from Reconstruct all slices."""
        self._set_source(array, array.shape[0], shape=array.shape if array.ndim == 3 else None, kind="volume")

    def show_stack_lazy(
        self, frame_getter: Callable[[object], np.ndarray], n_frames: int,
        shape: Optional[tuple] = None, kind: str = "volume",
    ) -> None:
        """
        Browse `n_frames` frames one at a time, via `frame_getter(index)`,
        without ever loading more than the single currently-displayed
        one. For raw projections, pass a bound
        DataInterface.get_projections -- for an hdf5-backed project
        that reads just that one h5py slice, for a tiff-backed one
        just that one file, regardless of storage_mode.

        `frame_getter` must also accept a tuple index such as
        `(slice(None), j)` if `shape` (the full 3D shape) is given --
        that is what enables the y / x buttons (sinograms, for
        `kind="projections"`). Without `shape`, only z is available.
        """
        self._set_source(frame_getter, n_frames, shape=shape, kind=kind)

    def show_image(self, array: np.ndarray) -> None:
        """
        A single 2D image, no slider -- a single-slice preview. If it has
        the same shape as the image already showing, the zoom and pan are
        kept, so repeated previews (e.g. while trying out centers of
        rotation) don't jump back to the full view each time.
        """
        keep_zoom = self._current_frame is not None and tuple(self._current_frame.shape) == tuple(array.shape)
        self._set_source(array[np.newaxis], 1, shape=None, kind="volume", keep_zoom=keep_zoom, single=True)

    def clear(self) -> None:
        self._source = None
        self._shape = None
        self._n_frames = 0
        self._current_frame = None
        self._view.clear()
        self._frame_slider.blockSignals(True)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.blockSignals(False)
        self._frame_label.setText("frame: -")
        self._readout.setText("x: -, y: -, value: -")
        self._set_axis(0, show=False)
        self._update_axis_buttons()

    def _set_source(
        self, source: FrameSource, n_frames: int, shape: Optional[tuple], kind: str,
        keep_zoom: bool = False, single: bool = False,
    ) -> None:
        self._source = source
        self._single = single
        self._shape = tuple(shape) if shape is not None and len(shape) == 3 else None
        self._kind = kind
        self._n_frames = n_frames
        self._set_axis(0, show=False)  # a new source always starts in the z view
        self._update_axis_buttons()
        self._configure_slider(n_frames, value=0)
        self._show_frame(0, reset_view=True, keep_zoom=keep_zoom)

    # -- remembering where a project was left ----------------------------------

    def view_state(self) -> Optional[dict]:
        """
        Slicing direction, slice number and zoom/pan of the stack on
        screen, to hand back to restore_view() later. None when nothing
        browsable is showing (empty, or a single preview image).
        """
        if self._source is None or self._single or self._current_frame is None:
            return None
        x_range, y_range = self._view.view.viewRange()
        return {
            "axis": self._axis,
            "index": self._frame_slider.value(),
            "frame_shape": tuple(self._current_frame.shape),
            "x_range": tuple(x_range),
            "y_range": tuple(y_range),
        }

    def restore_view(self, state: Optional[dict]) -> None:
        """Go back to a state from view_state(), after the same project's
        stack has been loaded again. The slice number is clamped to what
        this stack has; the zoom is only restored if the frames still
        have the same shape."""
        if not state or self._source is None or self._single:
            return
        axis = state["axis"] if (self._shape is not None or state["axis"] == 0) else 0
        n = self._shape[axis] if self._shape is not None else self._n_frames
        index = min(max(int(state["index"]), 0), max(n - 1, 0))
        if axis != self._axis or index != self._frame_slider.value():
            self._set_axis(axis, show=True)
            self._configure_slider(n, index)
            self._show_frame(index, reset_view=True)
        if tuple(self._current_frame.shape) == tuple(state["frame_shape"]):
            self._view.view.setRange(xRange=state["x_range"], yRange=state["y_range"], padding=0)

    # -- slicing axis ----------------------------------------------------------

    def axis(self) -> int:
        return self._axis

    def set_axis(self, axis: int) -> None:
        """Programmatic equivalent of clicking the z / y / x button."""
        if self._shape is None and axis != 0:
            return
        self._on_axis_clicked(axis)

    def _set_axis(self, axis: int, show: bool) -> None:
        self._axis = axis
        self._axis_buttons[axis].setChecked(True)  # group un-checks the others
        self._apply_crop_visibility()
        # A section containing the angle axis has no physical 1:1 pixel
        # scale (an angle step is not a pixel width), and angles often
        # outnumber pixels many times over (e.g. 2600 angles x 320
        # columns) -- locked 1:1 that is an unreadable thin strip.
        # Stretch it to fill the pane instead. Volume sections are
        # spatial on both axes, so they keep the true aspect ratio.
        locked = not (self._kind == "projections" and axis != 0)
        if locked != self._aspect_locked:  # setAspectLocked re-fits the view, so don't call it needlessly
            self._aspect_locked = locked
            self._view.view.setAspectLocked(locked)

    def _update_axis_buttons(self) -> None:
        sliceable = self._shape is not None
        for axis, button in enumerate(self._axis_buttons):
            button.setEnabled(sliceable or axis == 0)
            button.setToolTip(axis_tooltip(axis, self._kind))

    def _on_axis_clicked(self, axis: int) -> None:
        if self._source is None or (self._shape is None and axis != 0):
            self._axis_buttons[self._axis].setChecked(True)
            return
        self._set_axis(axis, show=True)
        n = self._shape[axis] if self._shape is not None else self._n_frames
        # Start at the middle section of a new direction: the first/last
        # section of a volume or sinogram stack is usually just background.
        self._configure_slider(n, value=n // 2 if axis else 0)
        self._show_frame(self._frame_slider.value(), reset_view=True)

    def _configure_slider(self, n_frames: int, value: int) -> None:
        self._frame_slider.blockSignals(True)
        self._frame_slider.setRange(0, max(n_frames - 1, 0))
        self._frame_slider.setValue(value)
        self._frame_slider.blockSignals(False)
        self._frame_slider.setVisible(n_frames > 1)
        self._frame_label.setVisible(n_frames > 1)

    def _frame_at(self, index: int) -> np.ndarray:
        key = section_index(self._axis, index)
        if isinstance(self._source, np.ndarray):
            return self._source[key]
        return self._source(key)  # a frame_getter callable, one section's worth of I/O

    def _show_frame(self, index: int, reset_view: bool, keep_zoom: bool = False) -> None:
        if self._source is None:
            return
        self._current_frame = self._frame_at(index)
        n = self._shape[self._axis] if self._shape is not None else self._n_frames
        self._frame_label.setText(f"{AXIS_NAMES[self._axis]}: {index} / {n - 1}")
        # Without an explicit axes= for a plain 2D array, pyqtgraph
        # assumes (x, y) order -- transposed from numpy's natural
        # (row, col) = (y, x). reset_view only on a fresh source (a
        # different project, or the first frame of a new
        # reconstruction) or a new slicing direction, so scrubbing the
        # slider preserves the user's current zoom/pan/contrast
        # instead of resetting it every frame.
        self._view.setImage(
            self._current_frame, axes={"y": 0, "x": 1}, autoLevels=reset_view, autoRange=reset_view and not keep_zoom,
        )

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
        if visible and self._current_frame is not None:
            height, width = self._current_frame.shape[-2], self._current_frame.shape[-1]
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

    def _on_mouse_moved(self, scene_pos) -> None:
        if self._current_frame is None:
            return

        view_pos = self._view.view.mapSceneToView(scene_pos)
        col, row = int(np.floor(view_pos.x())), int(np.floor(view_pos.y()))
        height, width = self._current_frame.shape[-2], self._current_frame.shape[-1]
        row_name, col_name = axis_labels(self._axis)

        if 0 <= row < height and 0 <= col < width:
            self._readout.setText(f"{col_name}: {col}, {row_name}: {row}, value: {self._current_frame[row, col]:.6g}")
        else:
            self._readout.setText(f"{col_name}: -, {row_name}: -, value: -")
