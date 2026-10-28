"""
The live line-profile window: the values of the image along the measurement line, redrawn
whenever the line is moved or another slice is shown. Opened with the K key or the
Profile button (see tools.py).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from imageviewer.measure import describe_line, line_profile


class LineProfileWindow(QWidget):
    def __init__(self, controller, parent=None):
        super().__init__(parent, Qt.Window)
        self.controller = controller
        self.canvas = controller.canvas
        self.setWindowTitle("Line profile")
        self.resize(560, 360)
        self.distance: Optional[np.ndarray] = None       # what is plotted, for tests and for Copy
        self.values: Optional[np.ndarray] = None

        self.plot = pg.PlotWidget(self)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.curve = self.plot.plot(pen=pg.mkPen(120, 200, 255, width=2))
        self.width_spin = QDoubleSpinBox(self)
        self.width_spin.setRange(1, 200)
        self.width_spin.setDecimals(0)
        self.width_spin.setValue(1)
        self.width_spin.setSuffix(" px wide")
        self.width_spin.setToolTip("Average this many parallel lines across the line's width (less noise).")
        self.copy_button = QPushButton("Copy data", self)
        self.copy_button.setToolTip("Copy distance and value as tab-separated text.")
        self.info_label = QLabel(self)
        self.stats_label = QLabel(self)

        row = QHBoxLayout()
        row.addWidget(self.info_label, 1)
        row.addWidget(self.width_spin)
        row.addWidget(self.copy_button)
        layout = QVBoxLayout(self)
        layout.addLayout(row)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.stats_label)

        self.width_spin.valueChanged.connect(lambda _v: self.update_profile())
        self.copy_button.clicked.connect(self.copy_data)
        controller.measurement_changed.connect(self.update_profile)
        self.canvas.frame_changed.connect(self.update_profile)
        self.update_profile()

    def update_profile(self) -> None:
        line = self.controller.current
        frame = self.canvas.frame
        if line is None or frame is None:
            self.distance = self.values = None
            self.curve.setData([], [])
            self.info_label.setText("Draw a line with the Line tool; this window follows it.")
            self.stats_label.setText("")
            return
        p0, p1 = line.points
        distance, values = line_profile(frame, p0, p1, width=self.width_spin.value())
        pixel = self.canvas.physical_pixel()
        self.distance, self.values = distance, values
        x = distance * pixel[0] if pixel else distance
        unit = pixel[1] if pixel else "px"
        self.curve.setData(x, values, connect="finite")
        self.plot.setLabel("bottom", f"distance ({unit})")
        self.plot.setLabel("left", "value")
        self.info_label.setText(describe_line(p0, p1, pixel))
        if np.isfinite(values).any():
            self.stats_label.setText(
                f"min {np.nanmin(values):.6g}   max {np.nanmax(values):.6g}   mean {np.nanmean(values):.6g}   "
                f"std {np.nanstd(values):.6g}   ({len(values)} samples)"
            )
        else:
            self.stats_label.setText("The line is outside the image.")

    def copy_data(self) -> None:
        if self.values is None:
            return
        pixel = self.canvas.physical_pixel()
        unit = pixel[1] if pixel else "px"
        scale = pixel[0] if pixel else 1.0
        lines = [f"distance_{unit}\tvalue"] + [f"{d * scale:.6g}\t{v:.6g}" for d, v in zip(self.distance, self.values)]
        QGuiApplication.clipboard().setText("\n".join(lines))
