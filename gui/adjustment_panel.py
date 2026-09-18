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
    QCheckBox,
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
from reconstruction import resolve_device

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

# Display text <-> ProjectConfig.device value. None ("Auto") is what
# every backend defaults to on its own (try GPU, fall back to CPU);
# see ReconstructionBackend.actual_device(). Order matters: it's the
# combo box's item order.
_DEVICE_CHOICES = [("Auto (prefer GPU)", None), ("CPU", "cpu"), ("GPU", "gpu")]


class BasicSettingsTab(QWidget):
    geometry_type_changed = Signal(str)
    angle_span_changed = Signal(float)
    pixel_size_changed = Signal(float)
    slice_changed = Signal(int)
    backend_changed = Signal(str)
    device_changed = Signal(object)  # None ("auto"), "cpu", or "gpu" -- see _DEVICE_CHOICES
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

        # voxel_size (core/known_fields.py's name for this; "pixel size"
        # here to match what the value actually measures for a
        # parallel-beam project -- detector_pixel_size, the value that
        # actually feeds reconstruct()'s unit conversion, is always
        # identical for parallel-beam, see derive_detector_pixel_size())
        # only ever has a "default" strategy, never "ask" -- it silently
        # becomes 1.0 rather than blocking anything, so this box exists
        # purely so a defaulted (i.e. unknown) value is visible and
        # fixable, not because completion would otherwise get stuck.
        self.pixel_size_spin = QDoubleSpinBox(self)
        self.pixel_size_spin.setRange(0.0001, 100000.0)
        self.pixel_size_spin.setDecimals(4)
        self.pixel_size_spin.setSuffix(" um")
        self.pixel_size_spin.valueChanged.connect(self.pixel_size_changed)

        self.slice_spin = QSpinBox(self)
        self.slice_spin.setMaximum(1_000_000)
        self.slice_spin.valueChanged.connect(self.slice_changed)

        self.backend_combo = QComboBox(self)
        # algotom_gridrec omitted from this list even though it's a
        # registered backend: it needs tomopy, which has no pip wheel
        # at all (conda-only) -- offering it here would just be a
        # button that always fails for anyone using this project's
        # normal pip install. Use it via the CLI/scripting API directly
        # if you've set up tomopy yourself.
        self.backend_combo.addItems([
            "algotom_fbp", "algotom_bpf", "algotom_dfi", "algotom_bp",
            "algotom_sirt", "algotom_sart", "algotom_art", "algotom_cgls",
        ])
        self.backend_combo.currentTextChanged.connect(self.backend_changed)

        self.device_combo = QComboBox(self)
        self.device_combo.addItems([label for label, _ in _DEVICE_CHOICES])
        self.device_combo.setToolTip(
            "Which device actually ran the last reconstruction is reported in the status "
            "message afterward -- a backend without its own GPU path always reports CPU, "
            "regardless of this setting."
        )
        self.device_combo.currentIndexChanged.connect(
            lambda index: self.device_changed.emit(_DEVICE_CHOICES[index][1])
        )

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
        form.addRow("Pixel size", self.pixel_size_spin)
        form.addRow("Slice (z)", self.slice_spin)
        form.addRow("Reconstruction algorithm", self.backend_combo)
        form.addRow("Device", self.device_combo)
        form.addRow("Center of rotation", self.center_spin)
        form.addRow("Tilt (deg)", self.tilt_spin)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(load_config_btn)
        layout.addStretch()

    def set_values(self, geometry_type: str, angle_span: float, angle_span_derived: bool, pixel_size: float, pixel_size_derived: bool, slice_index: int, slice_max: int, backend: str, device: Optional[str], center: float, tilt_deg: float, width: int) -> None:
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
        self._silently(self.pixel_size_spin, lambda: self.pixel_size_spin.setValue(pixel_size))
        # pixel_size_derived means this came from the file itself
        # (source == "file"); a silently-defaulted or previously
        # user-typed value stays editable -- same greyed-out-when-real
        # logic as angle_span_spin above, driven by source instead of
        # presence since voxel_size always resolves to *something*
        # (core/known_fields.py's default=1.0), never genuinely missing.
        self.pixel_size_spin.setEnabled(not pixel_size_derived)
        self.pixel_size_spin.setToolTip(
            "Read from this project's own recorded pixel size. Read-only."
            if pixel_size_derived
            else "Not recorded by this project's plugin -- silently defaulted to 1.0 um. The reconstruction's "
                 "1/cm unit is only physically meaningful once this is the real value; enter it if known."
        )
        self._silently(self.slice_spin, _set_slice)
        self._silently(self.backend_combo, lambda: self.backend_combo.setCurrentText(backend))
        device_index = next((i for i, (_, value) in enumerate(_DEVICE_CHOICES) if value == device), 0)
        self._silently(self.device_combo, lambda: self.device_combo.setCurrentIndex(device_index))
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
    """
    Everything here only ever affects the export path
    (MainWindow._reconstruct_selected(), "Reconstruct selected") --
    AdjustmentPanel's own two reconstruct buttons are preview-only and
    never postprocess or export, so none of this has any effect there.
    See doc/GUI_ARCHITECTURE.md's "Save settings" section.
    """

    changed = Signal()
    clip_min_changed = Signal(float)
    clip_max_changed = Signal(float)
    use_preview_range_requested = Signal()
    rows_from_changed = Signal(int)
    rows_to_changed = Signal(int)
    crop_toggled = Signal(bool)
    crop_rect_edited = Signal(tuple)  # (y0, y1, x0, x1), typed in rather than dragged

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dtype_combo = QComboBox(self)
        self.dtype_combo.addItems(["float32", "uint16"])
        self.dtype_combo.currentTextChanged.connect(self.changed)
        self.dtype_combo.currentTextChanged.connect(self._on_dtype_changed)

        self.format_combo = QComboBox(self)
        self.format_combo.addItems(["hdf5", "tiff"])
        self.format_combo.currentTextChanged.connect(self.changed)

        # Only meaningful once dtype == "uint16" -- to_uint16() clips to
        # [clip_min, clip_max] then rescales that range to the full
        # uint16 span; disabled otherwise (float32 export has no
        # clipping step at all). Range is deliberately wide open: these
        # hold real attenuation values (1/cm, see reconstruction/api.py),
        # not pixel indices.
        self.clip_min_spin = QDoubleSpinBox(self)
        self.clip_min_spin.setRange(-1.0e9, 1.0e9)
        self.clip_min_spin.setDecimals(6)
        self.clip_min_spin.valueChanged.connect(self.clip_min_changed)

        self.clip_max_spin = QDoubleSpinBox(self)
        self.clip_max_spin.setRange(-1.0e9, 1.0e9)
        self.clip_max_spin.setDecimals(6)
        self.clip_max_spin.valueChanged.connect(self.clip_max_changed)

        # The only cheap min/max available before an export actually
        # runs -- a full/batch export's own volume is never materialized
        # in the GUI process at all (run_batch() builds and exports it
        # entirely internally); computing a true global min/max would
        # mean reconstructing the whole thing just to find clip bounds.
        self.use_preview_range_btn = QPushButton("Use preview slice's range", self)
        self.use_preview_range_btn.clicked.connect(self.use_preview_range_requested)

        # Both INCLUSIVE ("slices 100 to 500" means literally that);
        # AdjustmentPanel/MainWindow convert to Python's half-open
        # convention when actually building config.rows.
        self.rows_from_spin = QSpinBox(self)
        self.rows_from_spin.setMaximum(1_000_000)
        self.rows_from_spin.valueChanged.connect(self.rows_from_changed)

        self.rows_to_spin = QSpinBox(self)
        self.rows_to_spin.setMaximum(1_000_000)
        self.rows_to_spin.valueChanged.connect(self.rows_to_changed)

        self.crop_checkbox = QCheckBox("Crop (drag the box in the viewer, or type below)", self)
        self.crop_checkbox.setToolTip(
            "Coordinates match whatever the viewer is currently showing -- reconstruct a slice preview "
            "first (a raw, un-reconstructed projection is a different, usually non-square, shape)."
        )
        self.crop_checkbox.toggled.connect(self.crop_toggled)
        self.crop_checkbox.toggled.connect(self._on_crop_toggled_enabled)

        # Both directions: dragging the viewer's crop box updates these
        # (AdjustmentPanel._on_crop_rect_changed() -> set_crop_spins()
        # below), and typing into these moves the box the same way a
        # drag would (crop_rect_edited -> AdjustmentPanel._on_crop_rect_edited()
        # -> ImageViewer.set_crop_rect()) -- so either one is always an
        # accurate, live readout of the other, useful on its own for a
        # reproducible/typed-in crop, not just as a readout of a drag.
        self.crop_y0_spin = QSpinBox(self)
        self.crop_y1_spin = QSpinBox(self)
        self.crop_x0_spin = QSpinBox(self)
        self.crop_x1_spin = QSpinBox(self)
        self._crop_spins = (self.crop_y0_spin, self.crop_y1_spin, self.crop_x0_spin, self.crop_x1_spin)
        for spin in self._crop_spins:
            spin.setMaximum(1_000_000)
            spin.valueChanged.connect(self._on_crop_spin_changed)

        start_row = QHBoxLayout()
        start_row.addWidget(QLabel("x:", self))
        start_row.addWidget(self.crop_x0_spin)
        start_row.addWidget(QLabel("y:", self))
        start_row.addWidget(self.crop_y0_spin)

        end_row = QHBoxLayout()
        end_row.addWidget(QLabel("x:", self))
        end_row.addWidget(self.crop_x1_spin)
        end_row.addWidget(QLabel("y:", self))
        end_row.addWidget(self.crop_y1_spin)

        form = QFormLayout()
        form.addRow("Output dtype", self.dtype_combo)
        form.addRow("Export format", self.format_combo)
        form.addRow("Clip min", self.clip_min_spin)
        form.addRow("Clip max", self.clip_max_spin)
        form.addRow("", self.use_preview_range_btn)
        form.addRow("From slice", self.rows_from_spin)
        form.addRow("To slice", self.rows_to_spin)
        form.addRow("", self.crop_checkbox)
        form.addRow("Starting point", start_row)
        form.addRow("End point", end_row)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("Rotation: not implemented yet.", self))
        layout.addStretch()

        self._on_dtype_changed(self.dtype_combo.currentText())

    def _on_dtype_changed(self, dtype: str) -> None:
        enabled = dtype == "uint16"
        self.clip_min_spin.setEnabled(enabled)
        self.clip_max_spin.setEnabled(enabled)
        self.use_preview_range_btn.setEnabled(enabled)

    def _on_crop_spin_changed(self, _value: int) -> None:
        y0, y1, x0, x1 = (spin.value() for spin in self._crop_spins)
        self.crop_rect_edited.emit((y0, y1, x0, x1))

    def set_crop_spins(self, rect: Optional[tuple[int, int, int, int]]) -> None:
        """Silently reflects a drag (or a restored session) into the
        four fields, without re-emitting crop_rect_edited -- that
        signal means "the user just typed a new value", not "a value
        arrived from elsewhere"."""
        y0, y1, x0, x1 = rect if rect is not None else (0, 0, 0, 0)
        for spin, value in zip(self._crop_spins, (y0, y1, x0, x1)):
            self._silently(spin, lambda spin=spin, value=value: spin.setValue(value))

    def set_values(
        self,
        dtype: str,
        export_format: str,
        clip_min: Optional[float],
        clip_max: Optional[float],
        rows_from: int,
        rows_to: int,
        rows_max: int,
        crop_checked: bool,
        crop_rect: Optional[tuple[int, int, int, int]],
        width: int,
    ) -> None:
        self._silently(self.dtype_combo, lambda: self.dtype_combo.setCurrentText(dtype))
        self._on_dtype_changed(dtype)  # setCurrentText above was silent, so the enabled-state slot never fired
        self._silently(self.format_combo, lambda: self.format_combo.setCurrentText(export_format))
        self._silently(self.clip_min_spin, lambda: self.clip_min_spin.setValue(clip_min if clip_min is not None else 0.0))
        self._silently(self.clip_max_spin, lambda: self.clip_max_spin.setValue(clip_max if clip_max is not None else 0.0))

        # setMaximum() can itself clamp+emit -- both spin boxes' ranges
        # and values need to change together inside ONE blockSignals
        # scope covering both widgets, same reasoning as
        # BasicSettingsTab.set_values()'s _set_slice()/_set_center().
        self.rows_from_spin.blockSignals(True)
        self.rows_to_spin.blockSignals(True)
        self.rows_from_spin.setMaximum(max(rows_max, 0))
        self.rows_to_spin.setMaximum(max(rows_max, 0))
        self.rows_from_spin.setValue(rows_from)
        self.rows_to_spin.setValue(rows_to)
        self.rows_from_spin.blockSignals(False)
        self.rows_to_spin.blockSignals(False)

        self._silently(self.crop_checkbox, lambda: self.crop_checkbox.setChecked(crop_checked))
        self._on_crop_toggled_enabled(crop_checked)

        for spin in self._crop_spins:
            spin.blockSignals(True)
            spin.setMaximum(max(width, 0))
            spin.blockSignals(False)
        self.set_crop_spins(crop_rect)

    def _on_crop_toggled_enabled(self, checked: bool) -> None:
        for spin in self._crop_spins:
            spin.setEnabled(checked)

    def values(self) -> tuple[str, str]:
        return self.dtype_combo.currentText(), self.format_combo.currentText()

    @staticmethod
    def _silently(widget, fn) -> None:
        widget.blockSignals(True)
        fn()
        widget.blockSignals(False)


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
        self.basic_tab.pixel_size_changed.connect(self._on_pixel_size_changed)
        self.basic_tab.slice_changed.connect(self._on_slice_changed)
        self.basic_tab.backend_changed.connect(self._on_backend_changed)
        self.basic_tab.device_changed.connect(self._on_device_changed)
        self.basic_tab.center_changed.connect(self._on_center_changed)
        self.basic_tab.tilt_changed.connect(self._on_tilt_changed)
        self.basic_tab.load_config_requested.connect(self._load_config_file)
        self.auto_tab.find_cor_requested.connect(self._find_cor)
        self.auto_tab.find_tilt_requested.connect(self._find_tilt)
        self.save_tab.changed.connect(self._on_save_settings_changed)
        self.save_tab.clip_min_changed.connect(self._on_clip_min_changed)
        self.save_tab.clip_max_changed.connect(self._on_clip_max_changed)
        self.save_tab.use_preview_range_requested.connect(self._on_use_preview_range_requested)
        self.save_tab.rows_from_changed.connect(self._on_rows_from_changed)
        self.save_tab.rows_to_changed.connect(self._on_rows_to_changed)
        self.save_tab.crop_toggled.connect(self._on_crop_toggled)
        self.save_tab.crop_rect_edited.connect(self._on_crop_rect_edited)
        self.viewer.crop_rect_changed.connect(self._on_crop_rect_changed)

        # The last single-slice preview's own reconstructed array --
        # the one cheap thing "Use preview slice's range" can read a
        # min/max from without materializing a full/batch volume just
        # to find clip bounds. Not session-scoped: only ever read right
        # after a preview reconstruct on whichever session is currently
        # open, same as the viewer's own displayed frame.
        self._last_preview_volume: Optional[np.ndarray] = None

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
        pixel_size = 1.0  # core/known_fields.py's voxel_size default; overridden below if known
        pixel_size_derived = False  # True only when source == "file" -- a real, measured value
        if project is not None:
            di = DataInterface(project)
            _, n_rows, width = di.shape_of("projections")
            geometry_type = di.get_metadata("geometry_type") or ""
            resolved_span = di.get_metadata("angle_span")
            if resolved_span is not None:
                angle_span = resolved_span
                angle_span_derived = True
            resolved_pixel_size = di.get_metadata("voxel_size")
            if resolved_pixel_size is not None:
                pixel_size = resolved_pixel_size
                pixel_size_derived = di.get_metadata_source("voxel_size") == "file"
        if "angle_span" in config.completion_answers:
            angle_span = config.completion_answers["angle_span"]  # a value the user already typed wins over the file's
            angle_span_derived = False  # still an editable answer, not yet re-resolved from data
        if "voxel_size" in config.completion_answers:
            pixel_size = config.completion_answers["voxel_size"]
            pixel_size_derived = False

        if config.center is None and width:
            # No offset is the only defensible guess before Find COR has
            # run; an arbitrary absolute column (e.g. 0) is not.
            config.center = width / 2.0

        if session.preview_row is None and n_rows:
            session.preview_row = n_rows // 2

        if session.export_rows is None and n_rows:
            session.export_rows = (0, n_rows - 1)  # the full range, same "resolve once, on first knowledge" pattern

        self.basic_tab.set_values(
            geometry_type=geometry_type,
            angle_span=angle_span,
            angle_span_derived=angle_span_derived,
            pixel_size=pixel_size,
            pixel_size_derived=pixel_size_derived,
            slice_index=session.preview_row or 0,
            slice_max=n_rows - 1,
            backend=config.reconstruction_backend,
            device=config.device,
            center=config.center if config.center is not None else 0.0,
            tilt_deg=config.tilt_deg if config.tilt_deg is not None else 0.0,
            width=width,
        )
        rows_from, rows_to = session.export_rows or (0, max(n_rows - 1, 0))
        self.save_tab.set_values(
            dtype=config.dtype,
            export_format=config.export_format,
            clip_min=config.dtype_clip_min,
            clip_max=config.dtype_clip_max,
            rows_from=rows_from,
            rows_to=rows_to,
            rows_max=max(n_rows - 1, 0),
            crop_checked=session.export_crop_yx is not None,
            crop_rect=session.export_crop_yx,
            width=width,
        )
        self.viewer.set_crop_rect(session.export_crop_yx)

    # -- basic settings edits ------------------------------------------------

    def _on_geometry_type_changed(self, value: str) -> None:
        if self.session is None or not value:
            return
        self.session.config.completion_answers["geometry_type"] = value

    def _on_angle_span_changed(self, value: float) -> None:
        if self.session is not None:
            self.session.config.completion_answers["angle_span"] = value

    def _on_pixel_size_changed(self, value: float) -> None:
        if self.session is not None:
            self.session.config.completion_answers["voxel_size"] = value

    def _on_slice_changed(self, value: int) -> None:
        if self.session is not None:
            self.session.preview_row = value

    def _on_backend_changed(self, value: str) -> None:
        if self.session is not None and value:
            self.session.config.reconstruction_backend = value

    def _on_device_changed(self, value) -> None:
        if self.session is not None:
            self.session.config.device = value

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

    def _on_clip_min_changed(self, value: float) -> None:
        if self.session is not None:
            self.session.config.dtype_clip_min = value

    def _on_clip_max_changed(self, value: float) -> None:
        if self.session is not None:
            self.session.config.dtype_clip_max = value

    def _on_use_preview_range_requested(self) -> None:
        if self.session is None:
            return
        if self._last_preview_volume is None:
            self.status_message.emit("reconstruct a slice preview first -- nothing to use its range from yet")
            return
        low, high = float(self._last_preview_volume.min()), float(self._last_preview_volume.max())
        # Not silent: this is a new choice being made (from the preview
        # instead of typed by hand), same as any other edit -- should
        # write into config exactly like typing the numbers in would.
        self.save_tab.clip_min_spin.setValue(low)
        self.save_tab.clip_max_spin.setValue(high)
        self.status_message.emit(f"clip range set from preview slice: [{low:.6g}, {high:.6g}]")

    def _on_rows_from_changed(self, value: int) -> None:
        if self.session is not None and self.session.export_rows is not None:
            self.session.export_rows = (value, self.session.export_rows[1])

    def _on_rows_to_changed(self, value: int) -> None:
        if self.session is not None and self.session.export_rows is not None:
            self.session.export_rows = (self.session.export_rows[0], value)

    def _on_crop_toggled(self, checked: bool) -> None:
        if self.session is None:
            return
        if not checked:
            self.viewer.set_crop_rect(None)
            self.save_tab.set_crop_spins(None)
            self.session.export_crop_yx = None
            return
        if self.session.export_crop_yx is not None:
            # set_crop_rect() is silent (see its docstring), so the
            # spin boxes need syncing explicitly here -- unlike the
            # "fresh" branch below, whose set_crop_visible() DOES emit
            # crop_rect_changed, already wired to do this.
            self.viewer.set_crop_rect(self.session.export_crop_yx)  # restore this session's own stored rect
            self.save_tab.set_crop_spins(self.session.export_crop_yx)
        else:
            self.viewer.set_crop_visible(True)  # fresh -- seed the default 80% box, crop_rect_changed picks it up

    def _on_crop_rect_edited(self, rect: tuple) -> None:
        """The reverse direction from _on_crop_rect_changed() below:
        someone typed into the four spin boxes instead of dragging."""
        if self.session is None:
            return
        self.session.export_crop_yx = rect
        self.viewer.set_crop_rect(rect)  # move the box to match; silent, so no feedback loop back here

    def _on_crop_rect_changed(self, rect) -> None:
        if self.session is not None:
            self.session.export_crop_yx = rect
        self.save_tab.set_crop_spins(rect)  # live readout of the drag, per the user's request

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
                self._last_preview_volume = volume[0]  # see SaveSettingsTab's "Use preview slice's range"
            device_used = resolve_device(session.config.reconstruction_backend, session.config.device)
            message = f"reconstructed {session.project_id}: {volume.shape}, 1/cm ({device_used.upper()})"
            # The 1/cm unit is only physically meaningful if pixel size
            # is real, not the silent 1.0 um default -- see Basic
            # settings' pixel size field and reconstruction/api.py's
            # "Units" docstring. Flagged here too since this is the
            # moment the number this actually affects gets produced.
            if DataInterface(session.run.project).get_metadata_source("voxel_size") != "file":
                message += " -- pixel size not read from the file (see Basic settings); 1/cm scale may not be accurate"
            self.status_message.emit(message)
            self.session_changed.emit(session)  # picks up the new "reconstructed" status color

        def failed(exc: Exception) -> None:
            self.busy_changed.emit(False)
            self.reconstruct_slice_btn.setEnabled(True)
            self.reconstruct_all_btn.setEnabled(True)
            session.error = str(exc)
            self.status_message.emit(f"reconstruction failed: {exc}")

        self._workers.append(run_in_background(_run_pipeline_to_reconstruction, session, on_finished=done, on_failed=failed))
