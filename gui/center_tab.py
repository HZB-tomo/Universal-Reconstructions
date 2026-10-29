"""
AutomaticAdjustmentsTab: the center-of-rotation scan and the tilt search.

A scan evaluates one metric (calibration/metrics.py) at a grid of trial centers and draws the curve:
the x axis is the center (as a shift from the detector middle, or as a detector column), the y axis the
quality, 1 being the best trial whichever way the metric points. A slider and a draggable line on the
plot move the current center along the curve; the panel previews the slice at every stop, so the
choice can be made by looking at the picture. Qt-only; the computing is the panel's and the engine's.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from calibration import CenterScan, get_metric, image_center, list_metrics

SHIFT, COLUMN = "Shift from the middle", "Detector column"
DEFAULT_POINTS = 101
MAX_POINTS = 2001
FINE_STEP_PX = 0.5          # a grid coarser than this is flagged: the minimum may fall between two trials
_KIND_TAG = {"sinogram": "fast", "slice": "reconstructs each"}


class AutomaticAdjustmentsTab(QWidget):
    scan_requested = Signal(str, float, float, int)   # metric name, range start and end (shift in px), number of points
    auto_requested = Signal()                         # the quick one-click estimate
    scan_cancel_requested = Signal()
    center_scrubbed = Signal(float, bool)             # center, final (False while the slider is still being dragged)
    settings_changed = Signal()                       # metric, range or number of points edited
    find_tilt_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._width: Optional[int] = None
        self._scan: Optional[CenterScan] = None
        self._current: Optional[float] = None
        self._busy = False
        self._updating = False
        self.range_edited = False       # the range was typed by the person (not set by a scan or a zoom)

        self.metric_combo = QComboBox(self)
        for name in list_metrics():
            metric = get_metric(name)
            self.metric_combo.addItem(f"{metric.label}  [{_KIND_TAG[metric.kind]}]", name)
            self.metric_combo.setItemData(self.metric_combo.count() - 1, metric.description, Qt.ToolTipRole)
        self.metric_combo.currentIndexChanged.connect(self._on_metric_changed)

        self.axis_combo = QComboBox(self)
        self.axis_combo.addItems([SHIFT, COLUMN])
        self.axis_combo.setToolTip("Show centers as the distance from the middle of the detector, or as detector columns.")
        self.axis_combo.currentIndexChanged.connect(self._on_axis_changed)

        self.from_spin = self._range_spin("Start of the range of trial centers.")
        self.to_spin = self._range_spin("End of the range of trial centers.")
        self.points_spin = QSpinBox(self)
        self.points_spin.setRange(3, MAX_POINTS)
        self.points_spin.setValue(DEFAULT_POINTS)
        self.points_spin.setToolTip("How many trial centers to evaluate, evenly spread over the range.")
        for spin in (self.from_spin, self.to_spin, self.points_spin):
            spin.valueChanged.connect(self._on_range_edited)
        for spin in (self.from_spin, self.to_spin):
            spin.valueChanged.connect(self._mark_range_edited)
        self.hint_label = QLabel("", self)
        self.hint_label.setWordWrap(True)

        self.auto_btn = QPushButton("Auto-find center", self)
        self.auto_btn.setToolTip(
            "A quick estimate: Vo's search over a tenth of the detector width either side of its middle, or over "
            "the Range below if you have typed one. The plot is the curve the search evaluated. Use the scan "
            "to look closer or with another metric."
        )
        self.auto_btn.clicked.connect(self.auto_requested)
        self.scan_btn = QPushButton("Scan centers", self)
        self.scan_btn.clicked.connect(self._on_scan_clicked)
        self.zoom_btn = QPushButton("Zoom in on best", self)
        self.zoom_btn.setToolTip("Scan again over a range a tenth as wide, centered on the best center found.")
        self.zoom_btn.clicked.connect(self._zoom)
        self.recentre_btn = QPushButton("Range around current", self)
        self.recentre_btn.setToolTip("Move the range so that it is centered on the current center of rotation.")
        self.recentre_btn.clicked.connect(self._recentre)
        self.best_btn = QPushButton("Go to best", self)
        self.best_btn.setToolTip("Set the center of rotation to the best center of the last scan.")
        self.best_btn.clicked.connect(self._go_to_best)

        self.progress = QProgressBar(self)
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)

        self.plot = pg.PlotWidget(self)
        self.plot.setMinimumHeight(170)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setMenuEnabled(False)
        self.plot.setYRange(0.0, 1.05, padding=0)
        self.plot.setLabel("left", "quality (1 = best)")
        self.plot.setLabel("bottom", "Center shift from the detector middle (px)")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.curve_item = self.plot.plot([], [], pen=pg.mkPen("#4fa3ff", width=2))
        self.best_item = pg.ScatterPlotItem(size=11, brush=pg.mkBrush("#3ecf6e"), pen=pg.mkPen("w", width=1))
        self.plot.addItem(self.best_item)
        self.current_line = pg.InfiniteLine(
            pos=0, angle=90, movable=True, pen=pg.mkPen("#ffb000", width=2), hoverPen=pg.mkPen("#ffd060", width=3),
        )
        self.current_line.setVisible(False)
        self.plot.addItem(self.current_line)
        self.current_line.sigPositionChanged.connect(lambda: self._on_line_moved(final=False))
        self.current_line.sigPositionChangeFinished.connect(lambda: self._on_line_moved(final=True))
        self.plot.scene().sigMouseClicked.connect(self._on_plot_clicked)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setEnabled(False)
        self.slider.setToolTip("Move the center of rotation along the scanned range; the preview follows.")
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.center_label = QLabel("", self)
        self.note_label = QLabel("", self)
        self.note_label.setWordWrap(True)

        self.find_tilt_btn = QPushButton("Find tilt", self)
        self.find_tilt_btn.clicked.connect(self.find_tilt_requested)

        form = QFormLayout()
        form.addRow("Metric", self.metric_combo)
        range_row = QHBoxLayout()
        range_row.addWidget(self.from_spin)
        range_row.addWidget(QLabel("to", self))
        range_row.addWidget(self.to_spin)
        form.addRow("Range", range_row)
        form.addRow("Axis", self.axis_combo)
        form.addRow("Points", self.points_spin)
        buttons = QHBoxLayout()
        buttons.addWidget(self.scan_btn)
        buttons.addWidget(self.zoom_btn)
        buttons2 = QHBoxLayout()
        buttons2.addWidget(self.recentre_btn)
        buttons2.addWidget(self.best_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.auto_btn)
        layout.addLayout(form)
        layout.addWidget(self.hint_label)
        layout.addLayout(buttons)
        layout.addLayout(buttons2)
        layout.addWidget(self.progress)
        layout.addWidget(self.plot)
        layout.addWidget(self.slider)
        layout.addWidget(self.center_label)
        layout.addWidget(self.note_label)
        layout.addSpacing(8)
        layout.addWidget(self.find_tilt_btn)
        layout.addStretch()
        self._refresh_enabled()
        self._refresh_hint()

    def _range_spin(self, tip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(self)
        spin.setRange(-100000.0, 100000.0)
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        spin.setSuffix(" px")
        spin.setToolTip(tip)
        return spin

    # -- the state the panel keeps per project ------------------------------------------------------------

    def set_width(self, width: Optional[int]) -> None:
        self._width = int(width) if width else None
        if self._width and self.from_spin.value() == self.to_spin.value():       # nothing chosen yet
            lo, hi = self.default_range()
            self._updating = True                        # a default is not an edit: nothing to report or remember
            try:
                self.from_spin.setValue(lo + self._offset())
                self.to_spin.setValue(hi + self._offset())
            finally:
                self._updating = False
        self._refresh_enabled()

    def metric(self) -> str:
        return self.metric_combo.currentData()

    def shift_range(self) -> tuple:
        """(start, end) of the range as shifts from the detector middle, whatever the axis display."""
        offset = self._offset()
        return self.from_spin.value() - offset, self.to_spin.value() - offset

    def points(self) -> int:
        return self.points_spin.value()

    def set_state(
        self, metric: str, shift_range: tuple, points: int, scan: Optional[CenterScan], current: Optional[float],
        range_edited: bool = False,
    ) -> None:
        """Show a project's settings and its last scan, without emitting anything."""
        self.range_edited = range_edited
        self._updating = True
        try:
            i = self.metric_combo.findData(metric)
            if i >= 0:
                self.metric_combo.setCurrentIndex(i)
            offset = self._offset()
            self.from_spin.setValue(shift_range[0] + offset)
            self.to_spin.setValue(shift_range[1] + offset)
            self.points_spin.setValue(int(points))
            self._scan = None
            self._current = current
        finally:
            self._updating = False
        self.set_scan(scan, move_to_best=False)
        self._refresh_hint()
        self.note_label.setText("")

    def default_range(self) -> tuple:
        """A first range to look in: an eighth of the detector either side of its middle."""
        half = (self._width or 100) / 8.0
        return -half, half

    # -- the scan --------------------------------------------------------------------------------------------

    def set_scan(self, scan: Optional[CenterScan], move_to_best: bool = True) -> None:
        self._scan = scan
        self._updating = True
        try:
            if scan is None:
                self.curve_item.setData([], [])
                self.best_item.setData([], [])
                self.slider.setEnabled(False)
                self.current_line.setVisible(False)
            else:
                x = scan.centers + (0.0 if self._column_mode() else -image_center(self._width or 1))
                self.curve_item.setData(x, scan.quality)
                self.best_item.setData([scan.best + (0.0 if self._column_mode() else -image_center(self._width or 1))],
                                       [scan.quality_at(scan.best)])
                self.plot.setXRange(float(x[0]), float(x[-1]), padding=0.02)
                self.slider.setRange(0, len(scan.centers) - 1)
                self.slider.setEnabled(True)
                self.current_line.setBounds((float(x[0]), float(x[-1])))
                self.current_line.setVisible(True)
        finally:
            self._updating = False
        if scan is not None and move_to_best:
            self._current = scan.best
        self._sync_current()
        self._refresh_enabled()

    def scan(self) -> Optional[CenterScan]:
        return self._scan

    def set_current_center(self, center: Optional[float]) -> None:
        """The center of rotation is now `center` (changed elsewhere): follow it, without emitting."""
        self._current = center
        self._sync_current()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.scan_btn.setText("Cancel scan" if busy else "Scan centers")
        self.progress.setVisible(busy)
        if busy:
            self.progress.setRange(0, 0)
        self._refresh_enabled()

    def set_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)
        self.progress.setFormat(f"{done} / {total} trial centers")

    def set_note(self, text: str) -> None:
        self.note_label.setText(text)

    # -- internals -------------------------------------------------------------------------------------------

    def _column_mode(self) -> bool:
        return self.axis_combo.currentText() == COLUMN

    def _offset(self) -> float:
        return image_center(self._width) if self._column_mode() and self._width else 0.0

    def _display(self, center: float) -> float:
        return center if self._column_mode() else center - image_center(self._width or 1)

    def _sync_current(self) -> None:
        """Put the line, the slider and the label on the current center."""
        scan, c = self._scan, self._current
        self._updating = True
        try:
            if scan is not None and c is not None:
                clamped = float(np.clip(c, scan.centers[0], scan.centers[-1]))
                self.current_line.setValue(self._display(clamped))
                self.slider.setValue(int(np.argmin(np.abs(scan.centers - clamped))))
        finally:
            self._updating = False
        if c is None or self._width is None:
            self.center_label.setText("")
            return
        text = f"center of rotation {c:.2f}  (shift {c - image_center(self._width):+.2f} px)"
        if scan is not None:
            text += f"    quality {scan.quality_at(c):.2f}"
            if not scan.centers[0] <= c <= scan.centers[-1]:
                text += "    (outside the scanned range)"
        self.center_label.setText(text)

    def _refresh_enabled(self) -> None:
        ready = self._width is not None
        for w in (self.metric_combo, self.axis_combo, self.from_spin, self.to_spin, self.points_spin, self.recentre_btn):
            w.setEnabled(ready and not self._busy)
        self.scan_btn.setEnabled(ready)
        self.auto_btn.setEnabled(ready and not self._busy)
        self.zoom_btn.setEnabled(ready and not self._busy and self._scan is not None)
        self.best_btn.setEnabled(ready and not self._busy and self._scan is not None)

    def _refresh_hint(self) -> None:
        lo, hi = self.from_spin.value(), self.to_spin.value()
        n = self.points_spin.value()
        if hi <= lo:
            self.hint_label.setText("<font color='#e07070'>the range is empty</font>")
            return
        step = (hi - lo) / (n - 1)
        text = f"{n} trial centers, {step:.2f} px apart"
        if get_metric(self.metric()).kind == "slice":
            text += f"; {n} reconstructions"
        if step > FINE_STEP_PX:
            text += f"<br><font color='#d0a030'>coarser than {FINE_STEP_PX} px: zoom in afterwards, or add points</font>"
        self.hint_label.setText(text)

    def _on_metric_changed(self, _i: int) -> None:
        self._refresh_hint()
        if not self._updating:
            self.settings_changed.emit()

    def _mark_range_edited(self, _v) -> None:
        if not self._updating:
            self.range_edited = True

    def auto_window(self) -> Optional[tuple]:
        """The window, as shifts from the detector middle, for the quick estimate to search: the range boxes if
        the person has typed them, else None (the default, a tenth of the width either side of the middle)."""
        lo, hi = self.shift_range()
        return (lo, hi) if self.range_edited and hi > lo else None

    def _on_range_edited(self, _v) -> None:
        self._refresh_hint()
        if not self._updating:
            self.settings_changed.emit()

    def _on_axis_changed(self, _i: int) -> None:
        """The same range, shown on the other axis: shifts become columns (add the middle) or back."""
        self.plot.setLabel("bottom", "Detector column" if self._column_mode() else "Center shift from the detector middle (px)")
        if self._width is None:
            return
        delta = image_center(self._width) * (1.0 if self._column_mode() else -1.0)
        self._updating = True
        try:
            self.from_spin.setValue(self.from_spin.value() + delta)
            self.to_spin.setValue(self.to_spin.value() + delta)
        finally:
            self._updating = False
        self.set_scan(self._scan, move_to_best=False)

    def _on_scan_clicked(self) -> None:
        if self._busy:
            self.scan_cancel_requested.emit()
            return
        lo, hi = self.shift_range()
        if hi <= lo:
            self.note_label.setText("The range is empty: the end must be larger than the start.")
            return
        self.scan_requested.emit(self.metric(), lo, hi, self.points())

    def _zoom(self) -> None:
        if self._scan is None or self._width is None:
            return
        lo, hi = self.shift_range()
        half = max(1.0, 0.05 * (hi - lo))
        best_shift = self._scan.best - image_center(self._width)
        self._updating = True
        try:
            offset = self._offset()
            self.from_spin.setValue(best_shift - half + offset)
            self.to_spin.setValue(best_shift + half + offset)
        finally:
            self._updating = False
        self._refresh_hint()
        self.settings_changed.emit()
        self._on_scan_clicked()

    def _recentre(self) -> None:
        if self._current is None or self._width is None:
            return
        lo, hi = self.shift_range()
        half = (hi - lo) / 2.0
        shift = self._current - image_center(self._width)
        offset = self._offset()
        self.from_spin.setValue(shift - half + offset)
        self.to_spin.setValue(shift + half + offset)

    def _go_to_best(self) -> None:
        if self._scan is not None:
            self._current = self._scan.best
            self._sync_current()
            self.center_scrubbed.emit(self._scan.best, True)

    def _scrub(self, center: float, final: bool) -> None:
        self._current = center
        self._sync_current()
        self.center_scrubbed.emit(center, final)

    def _on_slider_changed(self, index: int) -> None:
        if self._updating or self._scan is None:
            return
        self._scrub(float(self._scan.centers[index]), final=not self.slider.isSliderDown())

    def _on_slider_released(self) -> None:
        if self._scan is not None:
            self._scrub(float(self._scan.centers[self.slider.value()]), final=True)

    def _on_line_moved(self, final: bool) -> None:
        if self._updating or self._scan is None or self._width is None:
            return
        x = self.current_line.value()
        center = x if self._column_mode() else x + image_center(self._width)
        center = float(np.clip(center, self._scan.centers[0], self._scan.centers[-1]))
        self._scrub(center, final)

    def _on_plot_clicked(self, event) -> None:
        if self._scan is None or self._width is None or event.button() != Qt.LeftButton:
            return
        if not self.plot.sceneBoundingRect().contains(event.scenePos()):
            return
        x = self.plot.plotItem.vb.mapSceneToView(event.scenePos()).x()
        center = x if self._column_mode() else x + image_center(self._width)
        self._scrub(float(np.clip(center, self._scan.centers[0], self._scan.centers[-1])), final=True)
