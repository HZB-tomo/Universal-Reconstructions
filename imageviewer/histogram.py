"""
The histogram panel: a full contrast control for an ImageCanvas, meant to be docked
under the image (it is an ordinary QWidget, so it can sit in a QDockWidget that can also
float away).

    HistogramPanel(canvas)

What it offers: the histogram of the whole image or of the part inside the current zoom
(linear or logarithmic counts), the black/white points as a draggable region and as
numbers, Auto (levels from percentiles, with an adjustable clip), Min-Max, Scale to view,
Lock (keep these levels while scrolling through a stack or zooming), a colormap, and
the statistics of what the histogram shows.

The numbers are computed by the Qt-free functions at the top (compute_histogram,
percentile_levels), so they can be tested and reused without a window. New features
belong in this file only; the canvas is reached through its public properties.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from imageviewer.canvas import ImageCanvas

MAX_SAMPLES = 2_000_000      # larger frames are sampled with a regular stride; the shape of a histogram does not need more
DEFAULT_CLIP_PERCENT = 0.1   # Auto ignores this share of the darkest and of the brightest pixels


@dataclass
class Histogram:
    counts: np.ndarray
    edges: np.ndarray        # len(counts) + 1
    minimum: float
    maximum: float
    mean: float
    std: float
    n: int                   # pixels counted (finite ones)


def _finite_samples(values: np.ndarray, max_samples: int = MAX_SAMPLES) -> np.ndarray:
    flat = np.asarray(values).ravel()
    if flat.size > max_samples:
        flat = flat[:: int(np.ceil(flat.size / max_samples))]
    return flat[np.isfinite(flat)] if flat.dtype.kind == "f" else flat


def compute_histogram(values: Optional[np.ndarray], bins: int = 256) -> Optional[Histogram]:
    """Histogram and statistics of the finite values; None if there are none."""
    if values is None:
        return None
    flat = _finite_samples(values)
    if flat.size == 0:
        return None
    lo, hi = float(flat.min()), float(flat.max())
    if hi <= lo:
        hi = lo + 1.0
    counts, edges = np.histogram(flat, bins=bins, range=(lo, hi))
    return Histogram(counts, edges, float(flat.min()), float(flat.max()), float(flat.mean()), float(flat.std()), int(flat.size))


def percentile_levels(values: Optional[np.ndarray], clip_percent: float = DEFAULT_CLIP_PERCENT) -> Optional[tuple]:
    """(low, high) leaving out `clip_percent` percent of the pixels at each end; 0 gives min and max."""
    if values is None:
        return None
    flat = _finite_samples(values)
    if flat.size == 0:
        return None
    clip = float(np.clip(clip_percent, 0.0, 49.0))
    lo, hi = np.percentile(flat, [clip, 100.0 - clip]) if clip > 0 else (flat.min(), flat.max())
    lo, hi = float(lo), float(hi)
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


class HistogramPanel(QWidget):
    def __init__(self, canvas: ImageCanvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self._histogram: Optional[Histogram] = None
        self._updating = False
        self._timer = QTimer(self)               # zooming fires many range changes; recompute once things settle
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self.refresh)

        # -- buttons
        self.auto_button = QPushButton("Auto", self)
        self.auto_button.setToolTip("Black and white points from the percentiles of the data (the clip setting).")
        self.minmax_button = QPushButton("Min-Max", self)
        self.minmax_button.setToolTip("Black point at the smallest value, white point at the largest.")
        self.view_button = QPushButton("Scale to view", self)
        self.view_button.setToolTip("Fit the black and white points to the part of the image you are looking at.")
        self.lock_box = QCheckBox("Lock", self)
        self.lock_box.setToolTip("Keep these black and white points when another slice or image is shown.")
        self.log_box = QCheckBox("Log counts", self)
        self.log_box.setChecked(True)
        self.source_box = QComboBox(self)
        self.source_box.addItems(["Whole image", "Visible area"])
        self.source_box.setToolTip("Which pixels the histogram and its statistics describe.")

        self.clip_spin = QDoubleSpinBox(self)
        self.clip_spin.setRange(0.0, 49.0)
        self.clip_spin.setDecimals(2)
        self.clip_spin.setSingleStep(0.1)
        self.clip_spin.setValue(DEFAULT_CLIP_PERCENT)
        self.clip_spin.setSuffix(" % clip")
        self.clip_spin.setToolTip("Auto and Scale to view ignore this share of the darkest and of the brightest pixels.")
        self.min_spin = self._level_spin("Black point: pixels at or below this value are black.")
        self.max_spin = self._level_spin("White point: pixels at or above this value are white.")
        self.map_box = QComboBox(self)
        self.map_box.addItems(list(ImageCanvas.COLORMAPS))
        self.invert_box = QCheckBox("Invert", self)

        # -- the plot
        self.plot = pg.PlotWidget(self)
        self.plot.setMinimumHeight(90)
        self.plot.hideButtons()
        self.plot.setMenuEnabled(False)
        self.plot.getPlotItem().getViewBox().setMouseEnabled(x=True, y=False)
        self.curve = pg.PlotCurveItem(fillLevel=0, brush=(120, 160, 220, 160), pen=pg.mkPen(150, 190, 250))
        self.plot.addItem(self.curve)
        self.region = pg.LinearRegionItem(values=(0, 1), brush=(255, 255, 255, 30))
        self.plot.addItem(self.region)
        self.stats_label = QLabel(self)
        self.stats_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        top = QHBoxLayout()
        for widget in (self.auto_button, self.minmax_button, self.view_button, self.lock_box, self.clip_spin):
            top.addWidget(widget)
        top.addStretch()
        top.addWidget(self.source_box)
        top.addWidget(self.log_box)
        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("Black", self))
        bottom.addWidget(self.min_spin)
        bottom.addWidget(QLabel("White", self))
        bottom.addWidget(self.max_spin)
        bottom.addStretch()
        bottom.addWidget(QLabel("Colormap", self))
        bottom.addWidget(self.map_box)
        bottom.addWidget(self.invert_box)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(top)
        layout.addWidget(self.plot, 1)
        layout.addLayout(bottom)
        layout.addWidget(self.stats_label)

        # -- wiring
        self.auto_button.clicked.connect(self.auto_scale)
        self.minmax_button.clicked.connect(self.scale_to_minmax)
        self.view_button.clicked.connect(self.scale_to_view)
        self.lock_box.toggled.connect(self._on_lock_toggled)
        self.log_box.toggled.connect(lambda _on: self._draw())
        self.source_box.currentIndexChanged.connect(lambda _i: self.refresh())
        self.min_spin.editingFinished.connect(self._on_spins_edited)
        self.max_spin.editingFinished.connect(self._on_spins_edited)
        self.region.sigRegionChanged.connect(self._on_region_dragged)
        self.map_box.currentTextChanged.connect(self._on_colormap_changed)
        self.invert_box.toggled.connect(lambda _on: self._on_colormap_changed())
        canvas.frame_changed.connect(self._timer.start)
        canvas.view_changed.connect(self._on_view_changed)
        canvas.levels_changed.connect(self._on_levels_changed)

        self.lock_box.setChecked(canvas.levels_locked)
        self.refresh()
        if canvas.levels is not None:
            self._on_levels_changed(*canvas.levels)

    def _level_spin(self, tooltip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(self)
        spin.setRange(-1e12, 1e12)
        spin.setDecimals(6)
        spin.setKeyboardTracking(False)
        spin.setToolTip(tooltip)
        spin.setMinimumWidth(110)
        return spin

    # -- the data on screen ---------------------------------------------------

    def source_values(self) -> Optional[np.ndarray]:
        return self.canvas.visible_frame() if self.source_box.currentIndex() == 1 else self.canvas.frame

    def refresh(self) -> None:
        """Recompute the histogram from the canvas and redraw."""
        self._histogram = compute_histogram(self.source_values())
        self._draw()

    def _on_view_changed(self) -> None:
        if self.source_box.currentIndex() == 1:        # only the visible-area histogram depends on the zoom
            self._timer.start()

    def _draw(self) -> None:
        h = self._histogram
        if h is None:
            self.curve.setData([0, 1], [0], stepMode="center")
            self.stats_label.setText("No data")
            return
        counts = np.log10(h.counts + 1.0) if self.log_box.isChecked() else h.counts.astype(float)
        self.curve.setData(h.edges, counts, stepMode="center")
        lo, hi = self.canvas.levels or (h.minimum, h.maximum)
        span_lo, span_hi = min(h.minimum, lo), max(h.maximum, hi)
        pad = 0.02 * (span_hi - span_lo or 1.0)
        self.plot.setXRange(span_lo - pad, span_hi + pad, padding=0)
        self.plot.setLabel("left", "log10 counts" if self.log_box.isChecked() else "counts")
        where = "visible area" if self.source_box.currentIndex() == 1 else "image"
        self.stats_label.setText(
            f"{where}: {h.n:,} pixels   min {h.minimum:.6g}   max {h.maximum:.6g}   mean {h.mean:.6g}   std {h.std:.6g}"
        )

    # -- levels: from the canvas to the controls ------------------------------------

    def _on_levels_changed(self, lo: float, hi: float) -> None:
        self._updating = True
        try:
            for spin, value in ((self.min_spin, lo), (self.max_spin, hi)):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
            self.region.setRegion((lo, hi))
        finally:
            self._updating = False
        self._draw()

    # -- levels: from the controls to the canvas ------------------------------------

    def _on_region_dragged(self) -> None:
        if self._updating:
            return
        lo, hi = self.region.getRegion()
        self.canvas.set_levels(lo, hi)

    def _on_spins_edited(self) -> None:
        self.canvas.set_levels(self.min_spin.value(), self.max_spin.value())

    def _apply(self, levels: Optional[tuple]) -> None:
        if levels is not None:
            self.canvas.set_levels(*levels)

    def auto_scale(self) -> None:
        self._apply(percentile_levels(self.canvas.frame, self.clip_spin.value()))

    def scale_to_minmax(self) -> None:
        self._apply(percentile_levels(self.canvas.frame, 0.0))

    def scale_to_view(self) -> None:
        self._apply(percentile_levels(self.canvas.visible_frame(), self.clip_spin.value()))

    def _on_lock_toggled(self, locked: bool) -> None:
        self.canvas.levels_locked = locked

    def _on_colormap_changed(self, *_args) -> None:
        self.canvas.set_colormap(self.map_box.currentText(), self.invert_box.isChecked())
