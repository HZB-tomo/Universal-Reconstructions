"""
StackViewer: browse a stack of images (or a single image) with a slider, slice it along z / y / x,
read values under the cursor, and draw on it. It puts together the pieces of this package:

    viewer.canvas            ImageCanvas       the picture, levels, colormap, zoom
    viewer.tools             ToolController    line tool, K profile
    viewer.make_toolbar()    QToolBar          the tools, dockable in a QMainWindow
    viewer.make_histogram()  HistogramPanel    contrast control, dockable under the image

The frames come from a numpy array, or lazily from a callable that returns one section at
a time (so a stack on disk is never loaded whole). z / y / x choose which way a 3D stack is
sliced (slicing.py); z -- the first axis -- is the default.
"""
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QLabel, QPushButton, QSlider, QToolBar, QVBoxLayout, QWidget

from imageviewer.canvas import ImageCanvas
from imageviewer.histogram import HistogramPanel
from imageviewer.slicing import AXIS_NAMES, axis_labels, axis_tooltip, section_index
from imageviewer.tools import ToolController

# A numpy array, or a callable taking a numpy-style index (an int, or a
# tuple such as (slice(None), j)) and returning that section.
FrameSource = Union[np.ndarray, Callable[[object], np.ndarray]]


class StackViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = ImageCanvas(self)
        self.tools = ToolController(self.canvas)

        self._source: Optional[FrameSource] = None
        self._shape: Optional[tuple] = None  # (n0, n1, n2) of a 3D source; None = not sliceable
        self._kind = "volume"  # or "projections": only changes the tooltips and the aspect of y / x sections
        self._axis = 0
        self._single = False  # showing one preview image (show_image), not a browsable stack
        self._aspect_locked = True
        self._n_frames = 0

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
        layout.addWidget(self.canvas, 1)
        layout.addLayout(frame_row)
        layout.addWidget(self._readout)

        self._update_axis_buttons()
        self.canvas.hovered.connect(self._on_hovered)

    # -- the other pieces, ready to dock -----------------------------------------------

    def make_toolbar(self, title: str = "Image tools") -> QToolBar:
        """The tool buttons; add the result to a QMainWindow (addToolBar) to make it dockable."""
        return self.tools.make_toolbar(title)

    def make_histogram(self) -> HistogramPanel:
        """The histogram panel for this viewer; put it in a QDockWidget under the image."""
        return HistogramPanel(self.canvas)

    def set_pixel_size(self, size: Optional[float], unit: str = "um") -> None:
        """Physical size of one pixel, so that measurements also read in `unit`. None = unknown."""
        self.canvas.pixel_size = float(size) if size else None
        self.canvas.pixel_unit = unit
        self.tools._refresh_label()

    # -- what to show --------------------------------------------------------------------

    def show_stack(self, array: np.ndarray) -> None:
        """A 3D array already fully in memory -- e.g. a reconstructed volume."""
        self._set_source(array, array.shape[0], shape=array.shape if array.ndim == 3 else None, kind="volume")

    def show_stack_lazy(
        self, frame_getter: Callable[[object], np.ndarray], n_frames: int,
        shape: Optional[tuple] = None, kind: str = "volume",
    ) -> None:
        """
        Browse `n_frames` frames one at a time, via `frame_getter(index)`, without ever loading more
        than the single currently-displayed one.

        `frame_getter` must also accept a tuple index such as `(slice(None), j)` if `shape` (the full
        3D shape) is given -- that is what enables the y / x buttons. Without `shape`, only z is
        available. `kind="projections"` marks a stack whose first axis is an angle: its y / x sections
        are not true-to-scale pictures, so they are stretched to fill the pane.
        """
        self._set_source(frame_getter, n_frames, shape=shape, kind=kind)

    def show_image(self, array: np.ndarray) -> None:
        """
        A single 2D image, no slider. If it has the same shape as the image already showing, the zoom
        and pan are kept, so repeated previews don't jump back to the full view each time.
        """
        current = self.canvas.frame
        keep_zoom = current is not None and tuple(current.shape) == tuple(array.shape)
        self._set_source(array[np.newaxis], 1, shape=None, kind="volume", keep_zoom=keep_zoom, single=True)

    def clear(self) -> None:
        self._source = None
        self._shape = None
        self._n_frames = 0
        self.canvas.clear()
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

    @property
    def _current_frame(self) -> Optional[np.ndarray]:
        return self.canvas.frame

    def view_state(self) -> Optional[dict]:
        """
        Slicing direction, slice number and zoom/pan of the stack on screen, to hand back to
        restore_view() later. None when nothing browsable is showing (empty, or a single image).
        """
        if self._source is None or self._single or self.canvas.frame is None:
            return None
        x_range, y_range = self.canvas.view.viewRange()
        return {
            "axis": self._axis,
            "index": self._frame_slider.value(),
            "frame_shape": tuple(self.canvas.frame.shape),
            "x_range": tuple(x_range),
            "y_range": tuple(y_range),
        }

    def restore_view(self, state: Optional[dict]) -> None:
        """Go back to a state from view_state(), after the same stack has been loaded again. The slice
        number is clamped to what this stack has; the zoom is only restored if the frames still have
        the same shape."""
        if not state or self._source is None or self._single:
            return
        axis = state["axis"] if (self._shape is not None or state["axis"] == 0) else 0
        n = self._shape[axis] if self._shape is not None else self._n_frames
        index = min(max(int(state["index"]), 0), max(n - 1, 0))
        if axis != self._axis or index != self._frame_slider.value():
            self._set_axis(axis, show=True)
            self._configure_slider(n, index)
            self._show_frame(index, reset_view=True)
        if tuple(self.canvas.frame.shape) == tuple(state["frame_shape"]):
            self.canvas.view.setRange(xRange=state["x_range"], yRange=state["y_range"], padding=0)

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
        self._axis_changed()
        # A section containing the angle axis has no physical 1:1 pixel scale (an angle step is not a
        # pixel width), and angles often outnumber pixels many times over (e.g. 2600 angles x 320
        # columns) -- locked 1:1 that is an unreadable thin strip. Stretch it to fill the pane instead.
        # Volume sections are spatial on both axes, so they keep the true aspect ratio.
        locked = not (self._kind == "projections" and axis != 0)
        self.canvas.scale_valid = locked              # a pixel is only a length where the aspect is true
        if locked != self._aspect_locked:  # setAspectLocked re-fits the view, so don't call it needlessly
            self._aspect_locked = locked
            self.canvas.view.setAspectLocked(locked)

    def _axis_changed(self) -> None:
        """Hook for subclasses: the slicing direction was just set (self._axis)."""

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
        # Start at the middle section of a new direction: the first/last section of a volume or
        # sinogram stack is usually just background.
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
        frame = self._frame_at(index)
        n = self._shape[self._axis] if self._shape is not None else self._n_frames
        self._frame_label.setText(f"{AXIS_NAMES[self._axis]}: {index} / {n - 1}")
        # Levels and zoom are reset only on a fresh source (a different project, the first frame of a
        # new reconstruction) or a new slicing direction, so scrubbing the slider preserves the user's
        # current zoom/pan/contrast instead of resetting it every frame.
        self.canvas.set_frame(frame, reset_levels=reset_view, reset_view=reset_view, keep_zoom=keep_zoom)

    def _on_hovered(self, info) -> None:
        row_name, col_name = axis_labels(self._axis)
        if info is None:
            self._readout.setText(f"{col_name}: -, {row_name}: -, value: -")
        else:
            col, row, value = info
            self._readout.setText(f"{col_name}: {col}, {row_name}: {row}, value: {value:.6g}")
