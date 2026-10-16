"""
ConvertDialog: converts TIFF project folders to HDF5 files, optionally
binned, using data_import.converter.convert_to_hdf5(). Opened from
Tools > Convert TIFF project to HDF5. Standalone: it shares no state
with the projects open in the main window.
"""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from data_import.converter import ConversionCancelled, convert_to_hdf5
from data_import.plugin import get_plugin, get_plugins
from gui.import_dialog import select_multiple_directories
from gui.workers import run_in_background

AUTO_PLUGIN = "Auto-detect"


def output_path_for(source: str, output_dir: str, bin_factor: int) -> Path:
    """`<output_dir or the source's parent>/<folder name>[_bin<N>].h5`."""
    source_path = Path(source)
    suffix = f"_bin{bin_factor}" if bin_factor > 1 else ""
    return Path(output_dir or source_path.parent) / f"{source_path.name}{suffix}.h5"


class ConvertDialog(QDialog):
    progress_changed = Signal(int, int, str)  # emitted from the worker thread

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Convert TIFF project to HDF5")
        self.resize(640, 560)
        self._cancel = threading.Event()
        self._running = False
        self._workers: list = []

        self.sources = QListWidget(self)
        self.sources.setSelectionMode(QAbstractItemView.ExtendedSelection)
        add_btn = QPushButton("Add folder(s)...", self)
        add_btn.clicked.connect(self._add_folders)
        remove_btn = QPushButton("Remove", self)
        remove_btn.clicked.connect(self._remove_selected)
        source_buttons = QHBoxLayout()
        source_buttons.addWidget(add_btn)
        source_buttons.addStretch()
        source_buttons.addWidget(remove_btn)

        self.output_dir = QLineEdit(self)
        self.output_dir.setPlaceholderText("Next to each source folder")
        browse_btn = QPushButton("Browse...", self)
        browse_btn.clicked.connect(self._browse_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir)
        output_row.addWidget(browse_btn)

        self.plugin_combo = QComboBox(self)
        self.plugin_combo.addItem(AUTO_PLUGIN)
        self.plugin_combo.addItems([p.name for p in get_plugins("tiff")])

        self.bin_spin = QSpinBox(self)
        self.bin_spin.setRange(1, 64)
        self.bin_spin.setToolTip("Applied to projections, flat fields and dark fields. 1 = no binning.")
        self.method_combo = QComboBox(self)
        self.method_combo.addItems(["mean", "sum"])
        self.dtype_combo = QComboBox(self)
        self.dtype_combo.addItems(["auto", "uint16", "float32"])
        self.overwrite_checkbox = QCheckBox("Overwrite existing files", self)

        form = QFormLayout()
        form.addRow("Output folder", output_row)
        form.addRow("Plugin", self.plugin_combo)
        form.addRow("Bin factor", self.bin_spin)
        form.addRow("Binning method", self.method_combo)
        form.addRow("Output dtype", self.dtype_combo)
        form.addRow("", self.overwrite_checkbox)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 1)
        self.progress_label = QLabel("", self)
        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)

        self.convert_btn = QPushButton("Convert", self)
        self.convert_btn.setDefault(True)
        self.convert_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("Stop", self)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._cancel.set)
        self.close_btn = QPushButton("Close", self)
        self.close_btn.clicked.connect(self.close)
        buttons = QHBoxLayout()
        buttons.addWidget(self.convert_btn)
        buttons.addWidget(self.stop_btn)
        buttons.addStretch()
        buttons.addWidget(self.close_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("TIFF project folders", self))
        layout.addWidget(self.sources)
        layout.addLayout(source_buttons)
        layout.addLayout(form)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log)
        layout.addLayout(buttons)

        self.progress_changed.connect(self._on_progress)

    # -- source list ------------------------------------------------------

    def _add_folders(self) -> None:
        staged = set(self.source_paths())
        for path in select_multiple_directories(self, "Select TIFF project folder(s)"):
            if path not in staged:
                self.sources.addItem(path)
                staged.add(path)

    def _remove_selected(self) -> None:
        for item in self.sources.selectedItems():
            self.sources.takeItem(self.sources.row(item))

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Output folder", self.output_dir.text())
        if path:
            self.output_dir.setText(path)

    def source_paths(self) -> list[str]:
        return [self.sources.item(i).text() for i in range(self.sources.count())]

    # -- running ------------------------------------------------------------

    def _set_running(self, running: bool) -> None:
        self._running = running
        self.convert_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.close_btn.setEnabled(not running)

    def _start(self) -> None:
        sources = self.source_paths()
        if not sources:
            self.log.appendPlainText("Add at least one project folder.")
            return
        bin_factor = self.bin_spin.value()
        plugin_name = self.plugin_combo.currentText()
        options = dict(
            plugin=None if plugin_name == AUTO_PLUGIN else get_plugin("tiff", plugin_name),
            bin_y=bin_factor,
            bin_x=bin_factor,
            method=self.method_combo.currentText(),
            dtype=self.dtype_combo.currentText(),
            overwrite=self.overwrite_checkbox.isChecked(),
        )
        jobs = [(s, output_path_for(s, self.output_dir.text().strip(), bin_factor)) for s in sources]

        self._cancel.clear()
        self._set_running(True)
        self.progress_bar.setRange(0, 0)  # busy until the first progress report

        def work():
            results = []
            for source, output in jobs:
                if self._cancel.is_set():
                    break
                try:
                    convert_to_hdf5(
                        source, output, format="tiff", progress=self.progress_changed.emit,
                        cancel=self._cancel.is_set, **options,
                    )
                    results.append((source, str(output), None))
                except ConversionCancelled:
                    results.append((source, str(output), "cancelled"))
                    break
                except Exception as e:  # noqa: BLE001 -- reported per project; the batch continues
                    results.append((source, str(output), str(e)))
            return results

        def done(results) -> None:
            self._set_running(False)
            self.progress_bar.setRange(0, 1)
            self.progress_label.setText("")
            for source, output, error in results:
                self.log.appendPlainText(f"{source}\n  -> {output}: {'ok' if error is None else error}")
            if not results:
                self.log.appendPlainText("Nothing converted.")

        def failed(exc: Exception) -> None:
            self._set_running(False)
            self.progress_bar.setRange(0, 1)
            self.log.appendPlainText(f"Failed: {exc}")

        self._workers.append(run_in_background(work, on_finished=done, on_failed=failed))

    def _on_progress(self, done: int, total: int, label: str) -> None:
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)
        self.progress_label.setText(f"{label}: {done}/{total}")

    def closeEvent(self, event) -> None:
        if self._running:
            event.ignore()  # a running conversion has to be stopped first
        else:
            super().closeEvent(event)
