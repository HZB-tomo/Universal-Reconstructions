"""
ImageViewer: shows raw projections, a sinogram, or a reconstructed
slice/stack. A thin wrapper over pyqtgraph's ImageView, which already
provides the histogram/levels control and ROI tools. Its own frame
slider is hidden and replaced by ours (below): a single slider/display
path is used for both an eager, fully-in-memory stack (a reconstructed
volume -- reconstruct() always returns the whole thing, nothing lazy
to preserve there) and a lazy one (raw projections; see
show_stack_lazy()), rather than juggling two.
"""
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

FrameSource = Union[np.ndarray, Callable[[int], np.ndarray]]


class ImageViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._view = pg.ImageView()
        self._view.ui.roiPlot.hide()  # replaced by our own slider below

        self._source: Optional[FrameSource] = None
        self._n_frames = 0
        self._current_frame: Optional[np.ndarray] = None  # the displayed frame, cached for the hover readout

        self._frame_label = QLabel("frame: -", self)
        self._frame_slider = QSlider(Qt.Horizontal, self)
        self._frame_slider.valueChanged.connect(lambda i: self._show_frame(i, reset_view=False))

        self._readout = QLabel("x: -, y: -, value: -", self)

        frame_row = QHBoxLayout()
        frame_row.addWidget(self._frame_label)
        frame_row.addWidget(self._frame_slider)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)
        layout.addLayout(frame_row)
        layout.addWidget(self._readout)

        self._view.scene.sigMouseMoved.connect(self._on_mouse_moved)

    def show_stack(self, array: np.ndarray) -> None:
        """A 3D array already fully in memory -- e.g. a reconstructed
        volume from Reconstruct all slices."""
        self._set_source(array, array.shape[0])

    def show_stack_lazy(self, frame_getter: Callable[[int], np.ndarray], n_frames: int) -> None:
        """
        Browse `n_frames` frames one at a time, via `frame_getter(i)`,
        without ever loading more than the single currently-displayed
        one. For raw projections, pass a bound
        DataInterface.get_projections -- for an hdf5-backed project
        that reads just that one h5py slice, for a tiff-backed one
        just that one file, regardless of storage_mode.
        """
        self._set_source(frame_getter, n_frames)

    def show_image(self, array: np.ndarray) -> None:
        """A single 2D image, no slider."""
        self._set_source(array[np.newaxis], 1)

    def clear(self) -> None:
        self._source = None
        self._n_frames = 0
        self._current_frame = None
        self._view.clear()
        self._frame_slider.blockSignals(True)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.blockSignals(False)
        self._frame_label.setText("frame: -")
        self._readout.setText("x: -, y: -, value: -")

    def _set_source(self, source: FrameSource, n_frames: int) -> None:
        self._source = source
        self._n_frames = n_frames
        self._frame_slider.blockSignals(True)
        self._frame_slider.setRange(0, max(n_frames - 1, 0))
        self._frame_slider.setValue(0)
        self._frame_slider.blockSignals(False)
        self._frame_slider.setVisible(n_frames > 1)
        self._frame_label.setVisible(n_frames > 1)
        self._show_frame(0, reset_view=True)

    def _frame_at(self, index: int) -> np.ndarray:
        if isinstance(self._source, np.ndarray):
            return self._source[index]
        return self._source(index)  # a frame_getter callable, one frame's worth of I/O

    def _show_frame(self, index: int, reset_view: bool) -> None:
        if self._source is None:
            return
        self._current_frame = self._frame_at(index)
        self._frame_label.setText(f"frame: {index} / {self._n_frames - 1}")
        # Without an explicit axes= for a plain 2D array, pyqtgraph
        # assumes (x, y) order -- transposed from numpy's natural
        # (row, col) = (y, x). reset_view only on a fresh source (a
        # different project, or the first frame of a new
        # reconstruction), so scrubbing the slider preserves the
        # user's current zoom/pan/contrast instead of resetting it
        # every frame.
        self._view.setImage(self._current_frame, axes={"y": 0, "x": 1}, autoLevels=reset_view, autoRange=reset_view)

    def _on_mouse_moved(self, scene_pos) -> None:
        if self._current_frame is None:
            return

        view_pos = self._view.view.mapSceneToView(scene_pos)
        col, row = int(np.floor(view_pos.x())), int(np.floor(view_pos.y()))
        height, width = self._current_frame.shape[-2], self._current_frame.shape[-1]

        if 0 <= row < height and 0 <= col < width:
            self._readout.setText(f"x: {col}, y: {row}, value: {self._current_frame[row, col]:.6g}")
        else:
            self._readout.setText("x: -, y: -, value: -")
