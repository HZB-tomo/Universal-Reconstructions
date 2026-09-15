"""
AdjustmentPanel: tabbed controls for the currently open project, plus
the two reconstruct actions, which sit outside the tabs so they're
reachable no matter which tab is active. See doc/GUI_ARCHITECTURE.md.

Tabs: Basic settings, Automatic adjustments, Refinements (not
implemented), Save settings.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from calibration import find_center, find_tilt
from core.data_interface import DataInterface
from orchestration import PipelineRun, load_config
from preprocessing import build_reference, normalize_slice

from gui import ram_check
from gui.session import ProjectSession
from gui.viewer import ImageViewer
from gui.workers import run_in_background

# Sanity bounds, not hard limits: a real center of rotation is within
# about 1-2% of the detector width from the image's geometric center,
# and rotation-axis tilt is within about 1 degree. Values outside these
# get flagged in the status message, not rejected.
MAX_PLAUSIBLE_COR_OFFSET_FRACTION = 0.02
MAX_PLAUSIBLE_TILT_DEG = 1.0


class BasicSettingsTab(QWidget):
    geometry_type_changed = Signal(str)
    angle_span_changed = Signal(float)
    slice_changed = Signal(int)
    backend_changed = Signal(str)
    center_changed = Signal(float)
    tilt_changed = Signal(float)
    load_config_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.geometry_combo = QComboBox(self)
        self.geometry_combo.addItems(["", "parallel", "cone"])
        self.geometry_combo.currentTextChanged.connect(self.geometry_type_changed)

        # Only the span (end - start) is what completion/derivers.py's
        # derive_angles() actually consumes -- angles are always built
        # as np.linspace(0, angle_span, n, endpoint=False), so an
        # absolute start angle would have no effect on the result. One
        # field, not a start/end pair, so there's nothing to silently
        # discard.
        self.angle_span_spin = QDoubleSpinBox(self)
        self.angle_span_spin.setRange(0.0, 360.0)
        self.angle_span_spin.setDecimals(2)
        self.angle_span_spin.setSuffix(" deg")
        self.angle_span_spin.valueChanged.connect(self.angle_span_changed)

        self.slice_spin = QSpinBox(self)
        self.slice_spin.setMaximum(1_000_000)
        self.slice_spin.valueChanged.connect(self.slice_changed)

        self.backend_combo = QComboBox(self)
        self.backend_combo.addItems(["algotom_fbp"])
        self.backend_combo.currentTextChanged.connect(self.backend_changed)

        self.center_spin = QDoubleSpinBox(self)
        self.center_spin.setRange(-100000.0, 100000.0)
        self.center_spin.setDecimals(2)
        self.center_spin.valueChanged.connect(self.center_changed)

        self.tilt_spin = QDoubleSpinBox(self)
        self.tilt_spin.setRange(-3.0, 3.0)  # real tilt is within about 1 degree; see MAX_PLAUSIBLE_TILT_DEG
        self.tilt_spin.setDecimals(3)
        self.tilt_spin.valueChanged.connect(self.tilt_changed)

        load_config_btn = QPushButton("Load config file...", self)
        load_config_btn.clicked.connect(self.load_config_requested)

        form = QFormLayout()
        form.addRow("Geometry type", self.geometry_combo)
        form.addRow("Angle span", self.angle_span_spin)
        form.addRow("Slice (z)", self.slice_spin)
        form.addRow("Reconstruction algorithm", self.backend_combo)
        form.addRow("Center of rotation", self.center_spin)
        form.addRow("Tilt (deg)", self.tilt_spin)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(load_config_btn)
        layout.addStretch()

    def set_values(self, geometry_type: str, angle_span: float, angle_span_derived: bool, slice_index: int, slice_max: int, backend: str, center: float, tilt_deg: float, width: int) -> None:
        # setMaximum()/setRange() below can themselves clamp the
        # widget's current (still the *previous* session's) value and
        # emit valueChanged synchronously -- by the time that fires,
        # self.session already points at the *new* session (set_session()
        # reassigns it before calling this), so an unguarded clamp here
        # would silently write the old, wrong, clamped number onto the
        # new session before the correct setValue() below ever runs.
        # Every widget's range and value change together inside one
        # blockSignals scope so no such intermediate emission can occur.
        def _set_slice():
            self.slice_spin.setMaximum(max(slice_max, 0))
            self.slice_spin.setValue(slice_index)

        # A real center of rotation sits close to the image's geometric
        # center (see MAX_PLAUSIBLE_COR_OFFSET_FRACTION); the range just
        # keeps the field from accepting values nowhere near plausible,
        # it isn't itself the sanity check.
        def _set_center():
            margin = max(width * 0.1, 10)
            self.center_spin.setRange(-margin, width + margin)
            self.center_spin.setValue(center)

        self._silently(self.geometry_combo, lambda: self.geometry_combo.setCurrentText(geometry_type))
        self._silently(self.angle_span_spin, lambda: self.angle_span_spin.setValue(angle_span))
        # angle_span_derived means the file's own recorded angles already
        # give an exact value (e.g. 179.55 deg for a 400-projection,
        # endpoint=False 180 deg scan -- correct, not a rounding bug: see
        # completion/derivers.py's derive_angle_span()) -- editing it here
        # would have no effect until a re-import anyway, and greying it out
        # says so instead of inviting a pointless edit.
        self.angle_span_spin.setEnabled(not angle_span_derived)
        self.angle_span_spin.setToolTip(
            "Derived from this project's own recorded per-projection angles (max - min). Read-only."
            if angle_span_derived
            else "Not recorded by this project's plugin. Enter the scan's total angular sweep so per-projection angles can be derived."
        )
        self._silently(self.slice_spin, _set_slice)
        self._silently(self.backend_combo, lambda: self.backend_combo.setCurrentText(backend))
        self._silently(self.center_spin, _set_center)
        self._silently(self.tilt_spin, lambda: self.tilt_spin.setValue(tilt_deg))

    def set_center(self, value: float) -> None:
        self._silently(self.center_spin, lambda: self.center_spin.setValue(value))

    def set_tilt(self, value: float) -> None:
        self._silently(self.tilt_spin, lambda: self.tilt_spin.setValue(value))

    @staticmethod
    def _silently(widget, fn) -> None:
        widget.blockSignals(True)
        fn()
        widget.blockSignals(False)


class AutomaticAdjustmentsTab(QWidget):
    find_cor_requested = Signal()
    find_tilt_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        find_cor_btn = QPushButton("Find center of rotation", self)
        find_cor_btn.clicked.connect(self.find_cor_requested)
        find_tilt_btn = QPushButton("Find tilt", self)
        find_tilt_btn.clicked.connect(self.find_tilt_requested)

        layout = QVBoxLayout(self)
        layout.addWidget(find_cor_btn)
        layout.addWidget(find_tilt_btn)
        layout.addStretch()


class RefinementsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        label = QLabel("Ring and zinger artifact removal. Not implemented yet.", self)
        label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(label)
        layout.addStretch()


class SaveSettingsTab(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dtype_combo = QComboBox(self)
        self.dtype_combo.addItems(["float32", "uint16"])
        self.dtype_combo.currentTextChanged.connect(self.changed)

        self.format_combo = QComboBox(self)
        self.format_combo.addItems(["hdf5", "tiff"])
        self.format_combo.currentTextChanged.connect(self.changed)

        form = QFormLayout()
        form.addRow("Output dtype", self.dtype_combo)
        form.addRow("Export format", self.format_combo)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("Clipping, cropping, rotation: not wired up yet.", self))
        layout.addStretch()

    def set_values(self, dtype: str, export_format: str) -> None:
        self.dtype_combo.blockSignals(True)
        self.dtype_combo.setCurrentText(dtype)
        self.dtype_combo.blockSignals(False)
        self.format_combo.blockSignals(True)
        self.format_combo.setCurrentText(export_format)
        self.format_combo.blockSignals(False)

    def values(self) -> tuple[str, str]:
        return self.dtype_combo.currentText(), self.format_combo.currentText()


def _run_pipeline_to_reconstruction(session: ProjectSession) -> np.ndarray:
    if session.run is None:
        session.run = PipelineRun(session.config)
    run = session.run
    if run.project is None:
        run.import_data()
    run.complete()
    run.preprocess()
    run.calibrate()
    return run.reconstruct()


class AdjustmentPanel(QWidget):
    status_message = Signal(str)
    busy_changed = Signal(bool)  # True when a background pipeline call starts, False when it ends
    session_changed = Signal(object)  # ProjectSession whose config just changed, for ProjectPanel.refresh()

    def __init__(self, viewer: ImageViewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.session: Optional[ProjectSession] = None
        self._workers: list = []

        self.basic_tab = BasicSettingsTab(self)
        self.auto_tab = AutomaticAdjustmentsTab(self)
        self.refinements_tab = RefinementsTab(self)
        self.save_tab = SaveSettingsTab(self)

        self.tabs = QTabWidget(self)
        self.tabs.addTab(self.basic_tab, "Basic settings")
        self.tabs.addTab(self.auto_tab, "Automatic adjustments")
        self.tabs.addTab(self.refinements_tab, "Refinements")
        self.tabs.addTab(self.save_tab, "Save settings")

        self.reconstruct_slice_btn = QPushButton("Reconstruct (slice)", self)
        self.reconstruct_all_btn = QPushButton("Reconstruct all slices", self)
        self.reconstruct_slice_btn.clicked.connect(lambda: self._reconstruct(all_slices=False))
        self.reconstruct_all_btn.clicked.connect(lambda: self._reconstruct(all_slices=True))

        recon_row = QHBoxLayout()
        recon_row.addWidget(self.reconstruct_slice_btn)
        recon_row.addWidget(self.reconstruct_all_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addLayout(recon_row)

        self.basic_tab.geometry_type_changed.connect(self._on_geometry_type_changed)
        self.basic_tab.angle_span_changed.connect(self._on_angle_span_changed)
        self.basic_tab.slice_changed.connect(self._on_slice_changed)
        self.basic_tab.backend_changed.connect(self._on_backend_changed)
        self.basic_tab.center_changed.connect(self._on_center_changed)
        self.basic_tab.tilt_changed.connect(self._on_tilt_changed)
        self.basic_tab.load_config_requested.connect(self._load_config_file)
        self.auto_tab.find_cor_requested.connect(self._find_cor)
        self.auto_tab.find_tilt_requested.connect(self._find_tilt)
        self.save_tab.changed.connect(self._on_save_settings_changed)

        self.set_session(None)

    # -- session binding ----------------------------------------------------

    def set_session(self, session: Optional[ProjectSession]) -> None:
        self.session = session
        enabled = session is not None
        self.tabs.setEnabled(enabled)
        self.reconstruct_slice_btn.setEnabled(enabled)
        self.reconstruct_all_btn.setEnabled(enabled)
        if session is not None:
            self._refresh_from_session()

    def _refresh_from_session(self) -> None:
        session = self.session
        config = session.config
        project = session.run.project if session.run is not None else None

        n_rows, width = 1, 0
        geometry_type = ""
        angle_span = 180.0  # the common case (a half-turn scan); overridden below if known
        angle_span_derived = False  # True once a real, already-known value is shown (read-only)
        if project is not None:
            di = DataInterface(project)
            _, n_rows, width = di.shape_of("projections")
            geometry_type = di.get_metadata("geometry_type") or ""
            resolved_span = di.get_metadata("angle_span")
            if resolved_span is not None:
                angle_span = resolved_span
                angle_span_derived = True
        if "angle_span" in config.completion_answers:
            angle_span = config.completion_answers["angle_span"]  # a value the user already typed wins over the file's
            angle_span_derived = False  # still an editable answer, not yet re-resolved from data

        if config.center is None and width:
            # No offset is the only defensible guess before Find COR has
            # run; an arbitrary absolute column (e.g. 0) is not.
            config.center = width / 2.0

        if session.preview_row is None and n_rows:
            session.preview_row = n_rows // 2

        self.basic_tab.set_values(
            geometry_type=geometry_type,
            angle_span=angle_span,
            angle_span_derived=angle_span_derived,
            slice_index=session.preview_row or 0,
            slice_max=n_rows - 1,
            backend=config.reconstruction_backend,
            center=config.center if config.center is not None else 0.0,
            tilt_deg=config.tilt_deg if config.tilt_deg is not None else 0.0,
            width=width,
        )
        self.save_tab.set_values(dtype=config.dtype, export_format=config.export_format)

    # -- basic settings edits ------------------------------------------------

    def _on_geometry_type_changed(self, value: str) -> None:
        if self.session is None or not value:
            return
        self.session.config.completion_answers["geometry_type"] = value

    def _on_angle_span_changed(self, value: float) -> None:
        if self.session is not None:
            self.session.config.completion_answers["angle_span"] = value

    def _on_slice_changed(self, value: int) -> None:
        if self.session is not None:
            self.session.preview_row = value

    def _on_backend_changed(self, value: str) -> None:
        if self.session is not None and value:
            self.session.config.reconstruction_backend = value

    def _on_center_changed(self, value: float) -> None:
        if self.session is not None:
            self.session.config.center = value
            self.session_changed.emit(self.session)

    def _on_tilt_changed(self, value: float) -> None:
        if self.session is not None:
            self.session.config.tilt_deg = value
            self.session_changed.emit(self.session)

    def _on_save_settings_changed(self) -> None:
        if self.session is None:
            return
        dtype, export_format = self.save_tab.values()
        self.session.config.dtype = dtype
        self.session.config.export_format = export_format

    def _load_config_file(self) -> None:
        if self.session is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load config file", "", "Config files (*.ini)")
        if not path:
            return
        try:
            loaded = load_config(path)
        except ValueError as e:
            self.status_message.emit(f"failed to load config: {e}")
            return
        if isinstance(loaded, dict):
            self.status_message.emit("that is a batch config; load a single-project config here")
            return
        self.session.config = replace(loaded, project_id=self.session.config.project_id)
        self.session.run = None  # config changed under it; start a fresh PipelineRun next time
        self._refresh_from_session()
        self.session_changed.emit(self.session)
        self.status_message.emit(f"loaded config from {path}")

    # -- automatic adjustments ------------------------------------------------

    def _find_cor(self) -> None:
        session = self.session
        if session is None or not session.imported:
            self.status_message.emit("open and import the project before calibrating")
            return

        def work() -> float:
            project = session.run.project
            reference = build_reference(project, reducer=session.config.reference_reducer, dark_offset=session.config.dark_offset)
            n_rows = DataInterface(project).shape_of("projections")[1]
            middle_row = n_rows // 2
            sinogram = normalize_slice(project, middle_row, reference)
            return find_center(sinogram, backend=session.config.calibration_backend)

        def done(center: float) -> None:
            self.busy_changed.emit(False)
            session.config.center = center
            self.basic_tab.set_center(center)
            width = DataInterface(session.run.project).shape_of("projections")[2]
            offset_fraction = abs(center - width / 2.0) / width
            message = f"found center of rotation: {center:.2f} ({offset_fraction:.1%} of width from image center)"
            if offset_fraction > MAX_PLAUSIBLE_COR_OFFSET_FRACTION:
                message += " -- further off-center than usual, double check"
            self.status_message.emit(message)
            self.session_changed.emit(session)
            self._reconstruct(all_slices=False)

        def failed(exc: Exception) -> None:
            self.busy_changed.emit(False)
            self.status_message.emit(f"find center of rotation failed: {exc}")

        self.busy_changed.emit(True)
        self._workers.append(run_in_background(work, on_finished=done, on_failed=failed))

    def _find_tilt(self) -> None:
        session = self.session
        if session is None or not session.imported:
            self.status_message.emit("open and import the project before calibrating")
            return

        def work():
            run = session.run
            run.config.rows = None
            run.preprocess()
            return find_tilt(
                run.normalized,
                backend=session.config.calibration_backend,
                center=session.config.center,
                num_rows=session.config.tilt_num_rows or 12,
                margin=session.config.tilt_margin or 0,
                outlier_tolerance=session.config.tilt_outlier_tolerance or 20.0,
            )

        def done(result) -> None:
            self.busy_changed.emit(False)
            session.config.tilt_deg = result.tilt_deg
            self.basic_tab.set_tilt(result.tilt_deg)
            message = f"found tilt: {result.tilt_deg:.3f} deg"
            if abs(result.tilt_deg) > MAX_PLAUSIBLE_TILT_DEG:
                message += " -- larger than usual, double check"
            self.status_message.emit(message)
            self.session_changed.emit(session)
            self._reconstruct(all_slices=False)

        def failed(exc: Exception) -> None:
            self.busy_changed.emit(False)
            self.status_message.emit(f"find tilt failed: {exc}")

        self.busy_changed.emit(True)
        self._workers.append(run_in_background(work, on_finished=done, on_failed=failed))

    # -- reconstruction -------------------------------------------------------

    def _reconstruct(self, all_slices: bool) -> None:
        session = self.session
        if session is None:
            return

        if all_slices:
            _, n_rows, width = DataInterface(session.run.project).shape_of("projections")
            required = ram_check.estimate_volume_bytes(n_rows, width)
            if not ram_check.fits_in_available_ram(required):
                self.status_message.emit(
                    f"reconstructing all {n_rows} slices needs roughly {ram_check.describe(required)} of RAM, "
                    "not enough available -- refusing to start. Streaming a reconstruction to disk without "
                    "holding the whole volume in memory isn't implemented yet."
                )
                return

        session.config.rows = None if all_slices else slice(session.preview_row, session.preview_row + 1)
        label = "all slices" if all_slices else f"slice {session.preview_row}"
        self.status_message.emit(f"reconstructing {label} for {session.project_id} ...")
        self.reconstruct_slice_btn.setEnabled(False)
        self.reconstruct_all_btn.setEnabled(False)
        self.busy_changed.emit(True)

        def done(volume: np.ndarray) -> None:
            self.busy_changed.emit(False)
            self.reconstruct_slice_btn.setEnabled(True)
            self.reconstruct_all_btn.setEnabled(True)
            session.error = None
            if all_slices:
                self.viewer.show_stack(volume)
            else:
                self.viewer.show_image(volume[0])
            self.status_message.emit(f"reconstructed {session.project_id}: {volume.shape}")
            self.session_changed.emit(session)  # picks up the new "reconstructed" status color

        def failed(exc: Exception) -> None:
            self.busy_changed.emit(False)
            self.reconstruct_slice_btn.setEnabled(True)
            self.reconstruct_all_btn.setEnabled(True)
            session.error = str(exc)
            self.status_message.emit(f"reconstruction failed: {exc}")

        self._workers.append(run_in_background(_run_pipeline_to_reconstruction, session, on_finished=done, on_failed=failed))
