"""
AdjustmentPanel: tabbed controls for the currently open project, plus
the two reconstruct actions, which sit outside the tabs so they're
reachable no matter which tab is active. See doc/GUI_ARCHITECTURE.md.

Tabs: Basic settings, Automatic adjustments, Refinements, Save
settings.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import copy
import threading

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from calibration import find_tilt, half_scan_slice, search_center
from core.data_interface import DataInterface
from orchestration import PipelineRun, ProgressEvent, load_config, save_config
from preprocessing import build_reference, normalize_slice
from reconstruction import get_backend, list_backends, resolve_device, to_uint16

from orchestration import ram_check
from gui.preview_engine import PreviewEngine
from gui.reconstruct_dialog import ReconstructDestinationDialog
from gui.reconstruct_plan import RememberedPaths, describe_error, describe_run, make_plan, rows_for_export, streaming_config
from gui.session import ProjectSession
from gui.viewer import ImageViewer
from gui.workers import ProgressRelay, run_in_background

# Sanity bounds, not hard limits: a real center of rotation is within
# about 1-2% of the detector width from the image's geometric center,
# and rotation-axis tilt is within about 1 degree. Values outside these
# get flagged in the status message, not rejected.
MAX_PLAUSIBLE_COR_OFFSET_FRACTION = 0.02
MAX_PLAUSIBLE_TILT_DEG = 1.0

# Default seeded into an iterative backend's num_iter the first time one
# is selected. Reasonable for SIRT/SART/CGLS (see tests/test_reconstruction.py);
# ART needs num_iter on the order of n_angles*width instead -- see
# reconstruction/backends/algotom_astra.py's module docstring -- so it
# gets its own, much larger default (seeded once width/angles are known;
# see _on_backend_changed()).
_DEFAULT_NUM_ITER = 50

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
    num_iter_changed = Signal(int)
    center_changed = Signal(float)
    auto_update_toggled = Signal(bool)
    tilt_changed = Signal(float)
    rotate_changed = Signal(float)
    load_config_requested = Signal()
    export_config_requested = Signal()
    export_config_as_requested = Signal()

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
        # Whatever is registered AND usable right now: a backend whose packages are not
        # installed (algotom_gridrec needs conda-only tomopy, the ASTRA ones need astra,
        # gridrec needs urct_gridrec) is simply not offered, and one a user adds
        # shows up here without touching this file. See reconstruction.list_backends().
        self.backend_combo.addItems(list_backends())
        self.backend_combo.currentTextChanged.connect(self.backend_changed)

        # Only meaningful for the iterative backends (SIRT/SART/ART/CGLS
        # -- see ReconstructionBackend.is_iterative); hidden for the rest,
        # which ignore it entirely. algotom's own default, used whenever
        # this is left unset, is num_iter=1 -- a single sweep, nowhere
        # near converged, which is why an iterative backend can otherwise
        # look "wrong" compared to FBP/gridrec/BPF/DFI: those are direct,
        # non-iterative inversions, always fully "converged" by
        # construction, so they have nothing comparable to tune.
        self.num_iter_spin = QSpinBox(self)
        self.num_iter_spin.setRange(1, 10_000_000)
        self.num_iter_spin.valueChanged.connect(self.num_iter_changed)
        self.num_iter_label = QLabel("Iterations", self)

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
        self.center_spin.setSingleStep(0.5)  # one mouse-wheel notch / arrow press
        self.center_spin.valueChanged.connect(self.center_changed)

        # With this on, every change of the center (mouse wheel, typing,
        # picking a candidate) previews the slice again after a short
        # pause; previews already computed come back instantly.
        self.auto_update_checkbox = QCheckBox("Auto-update preview", self)
        self.auto_update_checkbox.setToolTip(
            "Reconstruct the preview slice again whenever the center of rotation changes. "
            "Results are remembered, so going back to a center you already tried is instant, "
            "and nearby centers are computed in the background."
        )
        self.auto_update_checkbox.toggled.connect(self.auto_update_toggled)

        self.tilt_spin = QDoubleSpinBox(self)
        self.tilt_spin.setRange(-3.0, 3.0)  # real tilt is within about 1 degree; see MAX_PLAUSIBLE_TILT_DEG
        self.tilt_spin.setDecimals(3)
        self.tilt_spin.valueChanged.connect(self.tilt_changed)

        # ProjectConfig.rotate_deg: rotates the reconstructed volume about
        # the scan axis (every xy slice, uniformly) -- see
        # reconstruction/rotation.py. Unlike tilt_spin above, this isn't a
        # physical property of the scan to find; it's a requested output
        # rotation, so the full +/-360deg range is allowed.
        self.rotate_spin = QDoubleSpinBox(self)
        self.rotate_spin.setRange(-360.0, 360.0)
        self.rotate_spin.setDecimals(2)
        self.rotate_spin.setSuffix(" deg")
        self.rotate_spin.setToolTip(
            "Rotate the reconstructed volume about the scan axis. Exact (reorders the sinogram's "
            "angle axis) when the scan covers a full 180deg (parallel-beam) or 360deg period; "
            "otherwise falls back to rotating the reconstructed image -- see doc/RECONSTRUCTION.md's "
            "\"Rotation\" section."
        )
        self.rotate_spin.valueChanged.connect(self.rotate_changed)

        load_config_btn = QPushButton("Load config file...", self)
        load_config_btn.clicked.connect(self.load_config_requested)

        # orchestration.save_config() -- writes this project's current
        # settings as a .ini "recipe" (same format load_config_btn
        # reads). export_config_btn writes next to the project's own
        # data file with no prompt (repeatable, like Ctrl+S); "Save
        # config as..." opens a Save-As dialog for an explicit path.
        export_config_btn = QPushButton("Save config", self)
        export_config_btn.setToolTip("Writes this project's current settings as a .ini file next to its data file.")
        export_config_btn.clicked.connect(self.export_config_requested)

        export_config_as_btn = QPushButton("Save config as...", self)
        export_config_as_btn.clicked.connect(self.export_config_as_requested)

        config_file_row = QHBoxLayout()
        config_file_row.addWidget(load_config_btn)
        config_file_row.addWidget(export_config_btn)
        config_file_row.addWidget(export_config_as_btn)

        # Lets the user edit values the project's own file supplied (angle
        # span, pixel size), which are otherwise greyed out as read-only.
        # The edit is stored as a completion answer, which completion
        # applies over the file's value. (A project that records its own
        # per-projection angles keeps those angles.)
        self._angle_span_derived = False
        self._pixel_size_derived = False
        self.edit_loaded_checkbox = QCheckBox("Allow editing values loaded from the project", self)
        self.edit_loaded_checkbox.setToolTip(
            "Angle span and pixel size read from the project's file are read-only by default. "
            "Tick this to override them."
        )
        self.edit_loaded_checkbox.toggled.connect(self._apply_loaded_editable)

        form = QFormLayout()
        form.addRow("", self.edit_loaded_checkbox)
        form.addRow("Geometry type", self.geometry_combo)
        form.addRow("Angle span", self.angle_span_spin)
        form.addRow("Pixel size", self.pixel_size_spin)
        form.addRow("Slice (z)", self.slice_spin)
        form.addRow("Reconstruction algorithm", self.backend_combo)
        form.addRow(self.num_iter_label, self.num_iter_spin)
        form.addRow("Device", self.device_combo)
        form.addRow("Center of rotation", self.center_spin)
        form.addRow("", self.auto_update_checkbox)
        form.addRow("Tilt (deg)", self.tilt_spin)
        form.addRow("Rotate (deg)", self.rotate_spin)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(config_file_row)
        layout.addStretch()

    def _apply_loaded_editable(self, allowed: bool) -> None:
        self.angle_span_spin.setEnabled(not self._angle_span_derived or allowed)
        self.pixel_size_spin.setEnabled(not self._pixel_size_derived or allowed)

    def set_values(self, geometry_type: str, angle_span: float, angle_span_derived: bool, pixel_size: float, pixel_size_derived: bool, slice_index: int, slice_max: int, backend: str, num_iter: int, is_iterative: bool, device: Optional[str], center: float, tilt_deg: float, rotate_deg: float, width: int) -> None:
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
        self._angle_span_derived = angle_span_derived
        self.angle_span_spin.setEnabled(not angle_span_derived or self.edit_loaded_checkbox.isChecked())
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
        self._pixel_size_derived = pixel_size_derived
        self.pixel_size_spin.setEnabled(not pixel_size_derived or self.edit_loaded_checkbox.isChecked())
        self.pixel_size_spin.setToolTip(
            "Read from this project's own recorded pixel size. Read-only."
            if pixel_size_derived
            else "Not recorded by this project's plugin -- silently defaulted to 1.0 um. The reconstruction's "
                 "1/cm unit is only physically meaningful once this is the real value; enter it if known."
        )
        self._silently(self.slice_spin, _set_slice)
        self._silently(self.backend_combo, lambda: self.backend_combo.setCurrentText(backend))
        self._silently(self.num_iter_spin, lambda: self.num_iter_spin.setValue(num_iter))
        self.num_iter_spin.setVisible(is_iterative)
        self.num_iter_label.setVisible(is_iterative)
        self.num_iter_spin.setToolTip(
            "Full sweeps over every projection (algotom's own default, used whenever this is left "
            "unset, is 1 -- almost always too few for a real reconstruction). ART is the exception: "
            "its 'iterations' are individual ray updates, not sweeps, so it needs a MUCH larger value "
            "here, on the order of (number of projections) x (detector width) -- see doc/RECONSTRUCTION.md."
            if backend == "algotom_art"
            else "Full sweeps over every projection. algotom's own default, used whenever this is left "
                 "unset, is 1 -- almost always too few for a real reconstruction; 50 is a reasonable "
                 "starting point."
        )
        device_index = next((i for i, (_, value) in enumerate(_DEVICE_CHOICES) if value == device), 0)
        self._silently(self.device_combo, lambda: self.device_combo.setCurrentIndex(device_index))
        self._silently(self.center_spin, _set_center)
        self._silently(self.tilt_spin, lambda: self.tilt_spin.setValue(tilt_deg))
        self._silently(self.rotate_spin, lambda: self.rotate_spin.setValue(rotate_deg))

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
    """
    Center-of-rotation search and tilt. The search can run fully automatic,
    or be steered with a guess and a tolerance (search window = guess +/-
    tolerance); it always lists the best few candidates, so when the first
    answer is wrong the next ones are one click away.
    """

    find_cor_requested = Signal(object, object)  # guess (float | None), tolerance (float | None)
    candidate_chosen = Signal(float)
    use_current_center_requested = Signal()
    find_tilt_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # "auto" is the special value at the minimum: no guess / no tolerance.
        self.guess_spin = QDoubleSpinBox(self)
        self.guess_spin.setRange(-1.0, 100000.0)
        self.guess_spin.setDecimals(1)
        self.guess_spin.setSpecialValueText("auto")
        self.guess_spin.setValue(-1.0)
        self.guess_spin.setToolTip(
            "Where you think the center of rotation is, in detector columns. 'auto' searches the middle "
            "half of the detector. Use the arrows or type a value to set a guess."
        )

        self.tolerance_spin = QDoubleSpinBox(self)
        self.tolerance_spin.setRange(0.0, 100000.0)
        self.tolerance_spin.setDecimals(1)
        self.tolerance_spin.setSuffix(" px")
        self.tolerance_spin.setSpecialValueText("auto")
        self.tolerance_spin.setToolTip(
            "How far from the guess to look (+/-). 'auto' is a tenth of the detector width around a guess."
        )

        use_current_btn = QPushButton("Use current center", self)
        use_current_btn.setToolTip("Copy the center of rotation currently set in Basic settings into the guess.")
        use_current_btn.clicked.connect(self.use_current_center_requested)

        self.find_cor_btn = QPushButton("Find center of rotation", self)
        self.find_cor_btn.clicked.connect(self._emit_find_cor)

        self.candidates_list = QListWidget(self)
        self.candidates_list.setMaximumHeight(110)
        self.candidates_list.setToolTip("Click a candidate to use it and preview it.")
        self.candidates_list.itemClicked.connect(self._on_candidate_clicked)
        self.search_note = QLabel("", self)
        self.search_note.setWordWrap(True)

        find_tilt_btn = QPushButton("Find tilt", self)
        find_tilt_btn.clicked.connect(self.find_tilt_requested)

        form = QFormLayout()
        form.addRow("Guess", self.guess_spin)
        form.addRow("Search +/-", self.tolerance_spin)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(use_current_btn)
        layout.addWidget(self.find_cor_btn)
        layout.addWidget(QLabel("Candidates (best first):", self))
        layout.addWidget(self.candidates_list)
        layout.addWidget(self.search_note)
        layout.addSpacing(12)
        layout.addWidget(find_tilt_btn)
        layout.addStretch()

    def guess(self) -> Optional[float]:
        return None if self.guess_spin.value() < 0 else self.guess_spin.value()

    def tolerance(self) -> Optional[float]:
        return None if self.tolerance_spin.value() <= 0 else self.tolerance_spin.value()

    def _emit_find_cor(self) -> None:
        self.find_cor_requested.emit(self.guess(), self.tolerance())

    def _on_candidate_clicked(self, item: QListWidgetItem) -> None:
        self.candidate_chosen.emit(float(item.data(Qt.UserRole)))

    def set_guess(self, value: Optional[float]) -> None:
        self.guess_spin.setValue(-1.0 if value is None else value)

    def set_values(self, guess: Optional[float], tolerance: Optional[float], candidates: list, note: str = "") -> None:
        for spin, value in ((self.guess_spin, -1.0 if guess is None else guess), (self.tolerance_spin, tolerance or 0.0)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.set_candidates(candidates, note)

    def set_candidates(self, candidates: list, note: str = "") -> None:
        """`candidates`: [(center, score), ...] best first (score NaN = unknown)."""
        self.candidates_list.clear()
        for center, score in candidates:
            label = f"{center:.2f}" if score != score else f"{center:.2f}    match {score:.2f}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, float(center))
            self.candidates_list.addItem(item)
        if candidates:
            self.candidates_list.setCurrentRow(0)
        self.search_note.setText(note)


class RefinementsTab(QWidget):
    """
    preprocessing.remove_ring_artifacts()/remove_zinger() (see
    preprocessing/corrections.py), applied in PipelineRun.preprocess()
    and the streaming per-chunk loop -- like Basic settings, changes
    here affect the preview reconstruct buttons immediately, not just
    "Reconstruct selected".
    """

    projections_normalized_toggled = Signal(bool)
    ring_correction_toggled = Signal(bool)
    ring_method_changed = Signal(str)
    zinger_removal_toggled = Signal(bool)
    zinger_threshold_changed = Signal(float)
    zinger_size_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.normalized_checkbox = QCheckBox("Projections already flat/dark corrected", self)
        self.normalized_checkbox.setToolTip(
            "Tick if the projections were already normalized upstream (e.g. ILL neutron data).\n"
            "The flat/dark division is then skipped everywhere: COR/tilt search, previews and\n"
            "reconstruction. The data is only scaled by its air level."
        )
        self.normalized_checkbox.toggled.connect(self.projections_normalized_toggled)

        self.ring_checkbox = QCheckBox("Ring correction", self)
        self.ring_checkbox.toggled.connect(self.ring_correction_toggled)
        self.ring_checkbox.toggled.connect(self._on_ring_toggled)

        # preprocessing/corrections.py's RingMethod -- the algotom
        # stripe-removal variants this codebase wraps.
        self.ring_method_combo = QComboBox(self)
        self.ring_method_combo.addItems([
            "sorting", "filtering", "fitting", "normalization", "all",
            "large_stripe", "dead_stripe", "interpolation", "2d_filtering_sorting",
            "fft", "wavelet_fft", "regularization",
        ])
        self.ring_method_combo.setToolTip(
            "sorting: general-purpose, least aggressive.\n"
            "filtering: sorting plus a smoothing filter.\n"
            "fitting: polynomial fit per column -- smooth/broad stripes.\n"
            "normalization: column-mean based.\n"
            "all: algotom's own combined multi-method pass.\n"
            "large_stripe: high-contrast/large stripes specifically.\n"
            "dead_stripe: unresponsive or fluctuating detector columns.\n"
            "interpolation: large+dead-stripe detection, interpolated across.\n"
            "2d_filtering_sorting: 2D low-pass filter + sorting.\n"
            "fft: Fourier-domain stripe suppression.\n"
            "wavelet_fft: wavelet + FFT, handles mixed-width stripes, costlier.\n"
            "regularization: regularized (TV-like) column correction."
        )
        self.ring_method_combo.currentTextChanged.connect(self.ring_method_changed)

        self.zinger_checkbox = QCheckBox("Zinger removal", self)
        self.zinger_checkbox.toggled.connect(self.zinger_removal_toggled)
        self.zinger_checkbox.toggled.connect(self._on_zinger_toggled)

        self.zinger_threshold_spin = QDoubleSpinBox(self)
        self.zinger_threshold_spin.setRange(0.0, 10.0)
        self.zinger_threshold_spin.setDecimals(3)
        self.zinger_threshold_spin.setToolTip("Outlier detection threshold. algotom's own default is 0.08.")
        self.zinger_threshold_spin.valueChanged.connect(self.zinger_threshold_changed)

        self.zinger_size_spin = QSpinBox(self)
        self.zinger_size_spin.setRange(1, 50)
        self.zinger_size_spin.setToolTip("Median filter size used to detect outliers. algotom's own default is 2.")
        self.zinger_size_spin.valueChanged.connect(self.zinger_size_changed)

        form = QFormLayout()
        form.addRow("", self.normalized_checkbox)
        form.addRow("", self.ring_checkbox)
        form.addRow("Ring method", self.ring_method_combo)
        form.addRow("", self.zinger_checkbox)
        form.addRow("Zinger threshold", self.zinger_threshold_spin)
        form.addRow("Zinger size", self.zinger_size_spin)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addStretch()

        self._on_ring_toggled(False)
        self._on_zinger_toggled(False)

    def _on_ring_toggled(self, checked: bool) -> None:
        self.ring_method_combo.setEnabled(checked)

    def _on_zinger_toggled(self, checked: bool) -> None:
        self.zinger_threshold_spin.setEnabled(checked)
        self.zinger_size_spin.setEnabled(checked)

    def set_values(
        self, ring_correction: bool, ring_method: str, zinger_removal: bool, zinger_threshold: float, zinger_size: int,
        projections_normalized: bool = False,
    ) -> None:
        self._silently(self.normalized_checkbox, lambda: self.normalized_checkbox.setChecked(projections_normalized))
        self._silently(self.ring_checkbox, lambda: self.ring_checkbox.setChecked(ring_correction))
        self._on_ring_toggled(ring_correction)
        self._silently(self.ring_method_combo, lambda: self.ring_method_combo.setCurrentText(ring_method))
        self._silently(self.zinger_checkbox, lambda: self.zinger_checkbox.setChecked(zinger_removal))
        self._on_zinger_toggled(zinger_removal)
        self._silently(self.zinger_threshold_spin, lambda: self.zinger_threshold_spin.setValue(zinger_threshold))
        self._silently(self.zinger_size_spin, lambda: self.zinger_size_spin.setValue(zinger_size))

    @staticmethod
    def _silently(widget, fn) -> None:
        widget.blockSignals(True)
        fn()
        widget.blockSignals(False)


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

        # Chunk-by-chunk reconstruction (PipelineRun._run_streaming(),
        # via run_batch()/_reconstruct_selected() below): only the
        # current chunk's data is ever in memory, not the whole
        # project's volume. Needs center already resolved (always true
        # here -- see AdjustmentPanel._refresh_from_session()) and, for
        # a uint16 export, clip_min/clip_max already resolved too (type
        # a value, or "Use preview slice's range" below) -- there's no
        # whole-volume min/max to fall back to a chunk at a time.
        self.stream_checkbox = QCheckBox("Stream to disk (low RAM)", self)
        self.stream_checkbox.setToolTip(
            "Reconstructs and writes one chunk of slices at a time instead of the whole "
            "volume at once -- for a project too large to fit in RAM as a single "
            "reconstruction. For a uint16 export, set Clip min/max explicitly first "
            "(typed, or 'Use preview slice's range' below) -- there's no whole-volume "
            "min/max to discover a chunk at a time."
        )
        self.stream_checkbox.toggled.connect(self.changed)

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
        form.addRow("", self.stream_checkbox)
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
        stream: bool,
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
        self._silently(self.stream_checkbox, lambda: self.stream_checkbox.setChecked(stream))
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

    def values(self) -> tuple[str, str, bool]:
        return self.dtype_combo.currentText(), self.format_combo.currentText(), self.stream_checkbox.isChecked()

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
    volume_written = Signal(object)  # ProjectSession whose full volume was just streamed to session.volume_path

    def __init__(self, viewer: ImageViewer, parent=None, paths: Optional[RememberedPaths] = None):
        super().__init__(parent)
        self.viewer = viewer
        self.paths = paths if paths is not None else RememberedPaths()
        self._relay = ProgressRelay(self)
        self.progress_event = self._relay.event  # ProgressEvent from a running reconstruction, for the status box
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
        self.basic_tab.num_iter_changed.connect(self._on_num_iter_changed)
        self.basic_tab.device_changed.connect(self._on_device_changed)
        self.basic_tab.center_changed.connect(self._on_center_changed)
        self.basic_tab.tilt_changed.connect(self._on_tilt_changed)
        self.basic_tab.rotate_changed.connect(self._on_rotate_changed)
        self.refinements_tab.projections_normalized_toggled.connect(self._on_projections_normalized_toggled)
        self.refinements_tab.ring_correction_toggled.connect(self._on_ring_correction_toggled)
        self.refinements_tab.ring_method_changed.connect(self._on_ring_method_changed)
        self.refinements_tab.zinger_removal_toggled.connect(self._on_zinger_removal_toggled)
        self.refinements_tab.zinger_threshold_changed.connect(self._on_zinger_threshold_changed)
        self.refinements_tab.zinger_size_changed.connect(self._on_zinger_size_changed)
        self.basic_tab.load_config_requested.connect(self._load_config_file)
        self.basic_tab.export_config_requested.connect(self._export_config)
        self.basic_tab.export_config_as_requested.connect(self._export_config_as)
        self.auto_tab.find_cor_requested.connect(self._find_cor)
        self.auto_tab.candidate_chosen.connect(self._on_candidate_chosen)
        self.auto_tab.use_current_center_requested.connect(self._use_current_center_as_guess)
        self.auto_tab.guess_spin.valueChanged.connect(self._on_guess_changed)
        self.auto_tab.tolerance_spin.valueChanged.connect(self._on_tolerance_changed)
        self.basic_tab.auto_update_toggled.connect(self._on_auto_update_toggled)
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

        # Single-slice previews with memory (see gui/preview_engine.py): every
        # preview image is kept, so scrubbing the center back and forth is
        # instant, and neighbouring centers are computed in the background.
        self.engine = PreviewEngine()
        self._auto_update = False
        self._preview_inflight = False
        self._preview_pending = False
        self._prefetch_cancel = threading.Event()
        self._auto_update_timer = QTimer(self)
        self._auto_update_timer.setSingleShot(True)
        self._auto_update_timer.setInterval(120)  # lets a burst of wheel notches settle
        self._auto_update_timer.timeout.connect(lambda: self._reconstruct(all_slices=False))

        self.set_session(None)

    # -- session binding ----------------------------------------------------

    def set_session(self, session: Optional[ProjectSession]) -> None:
        self._prefetch_cancel.set()  # background previews belong to the previous project
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

        n_angles, n_rows, width = 0, 1, 0
        geometry_type = ""
        angle_span = 180.0  # the common case (a half-turn scan); overridden below if known
        angle_span_derived = False  # True once a real, already-known value is shown (read-only)
        pixel_size = 1.0  # core/known_fields.py's voxel_size default; overridden below if known
        pixel_size_derived = False  # True only when source == "file" -- a real, measured value
        if project is not None:
            di = DataInterface(project)
            n_angles, n_rows, width = di.shape_of("projections")
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

        if "num_iter" not in config.reconstruction_options and get_backend(config.reconstruction_backend).is_iterative and n_angles and width:
            # Seeded once, on first knowledge -- same pattern as center/
            # preview_row/export_rows above -- so an iterative backend
            # actually gets a real num_iter passed through to reconstruct()
            # instead of silently falling back to algotom's own default
            # of 1 (see BasicSettingsTab.num_iter_spin's docstring/tooltip).
            # ART's very different iteration semantics need a much larger
            # value than the others -- see reconstruction/backends/algotom_astra.py.
            config.reconstruction_options["num_iter"] = (
                n_angles * width if config.reconstruction_backend == "algotom_art" else _DEFAULT_NUM_ITER
            )

        self.basic_tab.set_values(
            geometry_type=geometry_type,
            angle_span=angle_span,
            angle_span_derived=angle_span_derived,
            pixel_size=pixel_size,
            pixel_size_derived=pixel_size_derived,
            slice_index=session.preview_row or 0,
            slice_max=n_rows - 1,
            backend=config.reconstruction_backend,
            num_iter=config.reconstruction_options.get("num_iter", _DEFAULT_NUM_ITER),
            is_iterative=get_backend(config.reconstruction_backend).is_iterative,
            device=config.device,
            center=config.center if config.center is not None else 0.0,
            tilt_deg=config.tilt_deg if config.tilt_deg is not None else 0.0,
            rotate_deg=config.rotate_deg if config.rotate_deg is not None else 0.0,
            width=width,
        )
        self.refinements_tab.set_values(
            ring_correction=config.ring_correction,
            ring_method=config.ring_method,
            zinger_removal=config.zinger_removal,
            zinger_threshold=config.zinger_threshold if config.zinger_threshold is not None else 0.08,
            zinger_size=config.zinger_size if config.zinger_size is not None else 2,
            projections_normalized=config.projections_normalized,
        )
        rows_from, rows_to = session.export_rows or (0, max(n_rows - 1, 0))
        self.save_tab.set_values(
            dtype=config.dtype,
            export_format=config.export_format,
            stream=config.stream,
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
        self.auto_tab.set_values(session.cor_guess, session.cor_tolerance, session.cor_candidates)

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
            # Re-syncs num_iter_spin's visibility/tooltip/value for the
            # new backend (seeding a first-time default if it's iterative
            # and none exists yet) -- see the "num_iter" seeding block in
            # _refresh_from_session(). Safe to re-run in full: every other
            # value it touches is either read straight back from config/
            # session or a "resolve once" field that's already resolved.
            self._refresh_from_session()

    def _on_num_iter_changed(self, value: int) -> None:
        if self.session is not None:
            self.session.config.reconstruction_options["num_iter"] = value

    def _on_device_changed(self, value) -> None:
        if self.session is not None:
            self.session.config.device = value

    def _on_center_changed(self, value: float) -> None:
        if self.session is not None:
            self.session.config.center = value
            self.session_changed.emit(self.session)
            if self._auto_update and self.session.imported:
                self._auto_update_timer.start()  # restarts the pause if the wheel is still turning

    def _on_auto_update_toggled(self, checked: bool) -> None:
        self._auto_update = checked
        if checked and self.session is not None and self.session.imported:
            self._reconstruct(all_slices=False)

    def _on_guess_changed(self, _value: float) -> None:
        if self.session is not None:
            self.session.cor_guess = self.auto_tab.guess()

    def _on_tolerance_changed(self, _value: float) -> None:
        if self.session is not None:
            self.session.cor_tolerance = self.auto_tab.tolerance()

    def _use_current_center_as_guess(self) -> None:
        if self.session is not None and self.session.config.center is not None:
            self.auto_tab.set_guess(round(self.session.config.center, 1))

    def _on_candidate_chosen(self, center: float) -> None:
        """A candidate was clicked: make it the center (Basic settings
        follows) and preview it -- instantly if it was already computed."""
        if self.session is None:
            return
        self.session.config.center = center
        self.basic_tab.set_center(center)
        self.session_changed.emit(self.session)
        self._reconstruct(all_slices=False)

    def _on_tilt_changed(self, value: float) -> None:
        if self.session is not None:
            self.session.config.tilt_deg = value
            self.session_changed.emit(self.session)

    def _on_rotate_changed(self, value: float) -> None:
        if self.session is not None:
            self.session.config.rotate_deg = value
            self.session_changed.emit(self.session)

    # -- refinements edits ----------------------------------------------------

    def _on_projections_normalized_toggled(self, checked: bool) -> None:
        if self.session is not None:
            self.session.config.projections_normalized = checked

    def _on_ring_correction_toggled(self, checked: bool) -> None:
        if self.session is not None:
            self.session.config.ring_correction = checked

    def _on_ring_method_changed(self, value: str) -> None:
        if self.session is not None and value:
            self.session.config.ring_method = value

    def _on_zinger_removal_toggled(self, checked: bool) -> None:
        if self.session is not None:
            self.session.config.zinger_removal = checked

    def _on_zinger_threshold_changed(self, value: float) -> None:
        if self.session is not None:
            self.session.config.zinger_threshold = value

    def _on_zinger_size_changed(self, value: int) -> None:
        if self.session is not None:
            self.session.config.zinger_size = value

    def _on_save_settings_changed(self) -> None:
        if self.session is None:
            return
        dtype, export_format, stream = self.save_tab.values()
        self.session.config.dtype = dtype
        self.session.config.export_format = export_format
        self.session.config.stream = stream

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

    def _default_config_path(self) -> Path:
        return Path(self.session.config.path).parent / f"{self.session.project_id}_config.ini"

    def _export_config(self) -> None:
        if self.session is None:
            return
        path = self._default_config_path()
        save_config(self.session.config, path)
        self.status_message.emit(f"saved config to {path}")

    def _export_config_as(self) -> None:
        if self.session is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save config file", str(self._default_config_path()), "Config files (*.ini)")
        if not path:
            return
        save_config(self.session.config, path)
        self.status_message.emit(f"saved config to {path}")

    # -- automatic adjustments ------------------------------------------------

    def _find_cor(self, guess: Optional[float] = None, tolerance: Optional[float] = None) -> None:
        """
        Searches for the center of rotation on the middle row. `guess` and
        `tolerance` (both None = fully automatic) steer where it looks; the
        best few candidates are listed in the tab, the winner is applied and
        previewed, and the candidates are computed in the background so
        trying each of them afterwards is instant.
        """
        session = self.session
        if session is None or not session.imported:
            self.status_message.emit("open and import the project before calibrating")
            return

        def work():
            project = session.run.project
            reference = build_reference(
                project, reducer=session.config.reference_reducer, dark_offset=session.config.dark_offset,
                already_normalized=session.config.projections_normalized,
            )
            n_rows = DataInterface(project).shape_of("projections")[1]
            middle_row = n_rows // 2
            sinogram = normalize_slice(project, middle_row, reference)
            half = half_scan_slice(DataInterface(project).get_metadata("angles"))
            return search_center(
                sinogram[half], backend=session.config.calibration_backend, guess=guess, tolerance=tolerance,
            )

        def done(result) -> None:
            self.busy_changed.emit(False)
            self.auto_tab.find_cor_btn.setEnabled(True)
            session.cor_candidates = [(c.center, c.score) for c in result.candidates]
            session.config.center = result.best
            self.basic_tab.set_center(result.best)
            width = DataInterface(session.run.project).shape_of("projections")[2]
            offset_fraction = abs(result.best - width / 2.0) / width
            message = f"found center of rotation: {result.best:.2f} ({offset_fraction:.1%} of width from image center)"
            note = f"searched {result.start:.0f} to {result.stop:.0f}"
            if offset_fraction > MAX_PLAUSIBLE_COR_OFFSET_FRACTION:
                message += " -- further off-center than usual, double check"
            if result.at_edge:
                warning = (
                    "the best match is at the edge of the search window, so the true center may lie outside it -- "
                    "widen 'Search +/-' or change the guess"
                )
                message += " -- " + warning
                note += ". Warning: " + warning
            elif len(result.candidates) > 1 and result.candidates[1].score > 0.8:
                message += f" -- another candidate ({result.candidates[1].center:.1f}) scores almost as well, check the list"
            self.auto_tab.set_candidates(session.cor_candidates, note)
            self.status_message.emit(message)
            self.session_changed.emit(session)
            self._reconstruct(all_slices=False)
            self._schedule_prefetch([c.center for c in result.candidates[1:]])

        def failed(exc: Exception) -> None:
            self.busy_changed.emit(False)
            self.auto_tab.find_cor_btn.setEnabled(True)
            self.status_message.emit(f"find center of rotation failed: {exc}")

        self.busy_changed.emit(True)
        self.auto_tab.find_cor_btn.setEnabled(False)
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
            half = half_scan_slice(DataInterface(run.project).get_metadata("angles"))
            return find_tilt(
                run.normalized[half],
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

        if not all_slices:
            self._preview_slice()
            return

        if all_slices:
            if not session.imported:
                self.status_message.emit("open and import the project before reconstructing")
                return
            # Save settings' From/To slice range decides which slices "all
            # slices" means; set before asking so the RAM estimate, the
            # dialog and the (streamed or in-memory) run all agree on it.
            n_rows_total = DataInterface(session.run.project).shape_of("projections")[1]
            session.config.rows, n_selected = rows_for_export(session.export_rows, n_rows_total)
            if n_selected <= 0:
                self.status_message.emit("the From/To slice range in Save settings selects no slices")
                return
            path = self._ask_reconstruct_destination(session, n_rows_total, n_selected)
            if path == "":
                return  # cancelled, or refused (nothing fits and streaming isn't possible)
            if path is not None:
                self._reconstruct_streaming(session, path)
                return

        self.status_message.emit(f"reconstructing the selected slices for {session.project_id} ...")
        self.reconstruct_slice_btn.setEnabled(False)
        self.reconstruct_all_btn.setEnabled(False)
        self.busy_changed.emit(True)

        def done(volume: np.ndarray) -> None:
            self.busy_changed.emit(False)
            self.progress_event.emit(ProgressEvent(
                session.project_id, "Done", volume.shape[0], volume.shape[0],
                resolve_device(session.config.reconstruction_backend, session.config.device),
            ))
            self.reconstruct_slice_btn.setEnabled(True)
            self.reconstruct_all_btn.setEnabled(True)
            session.error = None
            session.ram_volume = volume  # shown through volume_written below, like a streamed volume
            session.volume_path = None  # the in-memory volume is now the latest full reconstruction
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
            self.volume_written.emit(session)

        def failed(exc: Exception) -> None:
            self.busy_changed.emit(False)
            self.reconstruct_slice_btn.setEnabled(True)
            self.reconstruct_all_btn.setEnabled(True)
            session.error = str(exc)
            self.status_message.emit(f"reconstruction failed: {describe_error(exc)}")
            self.progress_event.emit(ProgressEvent(session.project_id, "Failed", message=describe_error(exc)))
            if isinstance(exc, MemoryError):
                QMessageBox.warning(self, "Out of memory", describe_error(exc))

        def work() -> np.ndarray:
            if session.run is not None:
                session.run.progress = self._relay
            volume = _run_pipeline_to_reconstruction(session)
            if session.config.dtype == "uint16":
                # The full volume honours Save settings' output type too. With no
                # Clip min/max set, the volume's own min/max is used (it is all here).
                volume, _ = to_uint16(volume, session.config.dtype_clip_min, session.config.dtype_clip_max)
            return volume

        self._workers.append(run_in_background(work, on_finished=done, on_failed=failed))

    # -- single-slice preview (with memory) ----------------------------------------

    def _preview_slice(self) -> None:
        """Reconstructs the preview slice at the current center. A result the
        engine already has is shown at once; otherwise it is computed in the
        background. A request that arrives meanwhile (mouse wheel) is
        remembered and answered with the latest center when the current one
        finishes, so the view never lags behind a burst of changes."""
        session = self.session
        if session is None:
            return
        if not session.imported:
            self.status_message.emit("open and import the project before reconstructing")
            return
        self._auto_update_timer.stop()
        self._prefetch_cancel.set()  # the foreground request has priority over background work

        project = session.run.project
        config = copy.deepcopy(session.config)
        row = session.preview_row if session.preview_row is not None else 0
        center = config.center if config.center is not None else DataInterface(project).shape_of("projections")[2] / 2.0

        image = self.engine.cached(project, config, row, center)
        if image is not None:
            self._show_preview(session, image, center, cached=True)
            self._schedule_prefetch_neighbours(center)
            return
        if self._preview_inflight:
            self._preview_pending = True
            return

        self._preview_inflight = True
        self.reconstruct_slice_btn.setEnabled(False)
        self.busy_changed.emit(True)
        self.progress_event.emit(ProgressEvent(
            session.project_id, "Reconstructing", 0, 1, resolve_device(config.reconstruction_backend, config.device),
        ))

        def work() -> np.ndarray:
            session.run.complete()
            return self.engine.compute(project, config, row, center)

        def done(image: np.ndarray) -> None:
            self._preview_inflight = False
            self.busy_changed.emit(False)
            self.reconstruct_slice_btn.setEnabled(True)
            if self._preview_pending and self.session is session:
                self._preview_pending = False
                self._preview_slice()  # the center moved on while this one was computing
                return
            self._show_preview(session, image, center, cached=False)
            self._schedule_prefetch_neighbours(center)

        def failed(exc: Exception) -> None:
            self._preview_inflight = False
            self._preview_pending = False
            self.busy_changed.emit(False)
            self.reconstruct_slice_btn.setEnabled(True)
            session.error = str(exc)
            self.status_message.emit(f"reconstruction failed: {describe_error(exc)}")
            self.progress_event.emit(ProgressEvent(session.project_id, "Failed", message=describe_error(exc)))
            if isinstance(exc, MemoryError):
                QMessageBox.warning(self, "Out of memory", describe_error(exc))

        self._workers.append(run_in_background(work, on_finished=done, on_failed=failed))

    def _show_preview(self, session: ProjectSession, image: np.ndarray, center: float, cached: bool) -> None:
        if session is not self.session:
            return  # the user moved on to another project
        self.viewer.show_image(image)
        self._last_preview_volume = image  # see SaveSettingsTab's "Use preview slice's range"
        session.error = None
        device_used = resolve_device(session.config.reconstruction_backend, session.config.device)
        self.progress_event.emit(ProgressEvent(session.project_id, "Done", 1, 1, device_used, "preview"))
        message = (
            f"preview of {session.project_id}, slice {session.preview_row}, center {center:.2f}: "
            f"{image.shape}, 1/cm ({device_used.upper()})" + (" [remembered]" if cached else "")
        )
        # The 1/cm unit is only physically meaningful if pixel size
        # is real, not the silent 1.0 um default -- see Basic
        # settings' pixel size field and reconstruction/api.py's
        # "Units" docstring.
        if DataInterface(session.run.project).get_metadata_source("voxel_size") != "file":
            message += " -- pixel size not read from the file (see Basic settings); 1/cm scale may not be accurate"
        self.status_message.emit(message)
        self.session_changed.emit(session)

    PREFETCH_STEPS = 3  # neighbouring centers computed ahead, each side, in steps of the center box

    def _schedule_prefetch_neighbours(self, center: float) -> None:
        """With auto-update on, compute the centers a few mouse-wheel notches
        either side of this one in the background, nearest first."""
        if not self._auto_update:
            return
        step = self.basic_tab.center_spin.singleStep()
        centers = []
        for k in range(1, self.PREFETCH_STEPS + 1):
            centers += [center + k * step, center - k * step]
        self._schedule_prefetch(centers)

    def _schedule_prefetch(self, centers: list) -> None:
        session = self.session
        if session is None or not session.imported or not centers:
            return
        self._prefetch_cancel.set()  # replace any earlier batch
        cancel = threading.Event()
        self._prefetch_cancel = cancel
        project = session.run.project
        row = session.preview_row if session.preview_row is not None else 0
        run_in_background(
            self.engine.prefetch, project, session.config, row, list(centers), cancelled=cancel.is_set,
        )

    def _ask_reconstruct_destination(self, session: ProjectSession, n_rows_total: int, n_selected: int) -> Optional[str]:
        """
        Asks where the whole volume should go. Returns the HDF5 path to
        stream into, None to reconstruct in memory, or "" if the user
        cancelled or nothing is possible (a message has been shown).
        """
        di = DataInterface(session.run.project)
        n_angles, _, width = di.shape_of("projections")
        n_rows = n_selected
        plan = make_plan(n_angles, n_rows, width, di.get_metadata("geometry_type"))
        if not plan.can_stream and not plan.fits_in_ram:
            QMessageBox.warning(
                self, "Not enough memory",
                f"Reconstructing all {n_rows} slices needs about {ram_check.describe(plan.needed_bytes)} of RAM "
                f"but only {ram_check.describe(plan.available_bytes)} is available, and {plan.stream_blocker}.",
            )
            return ""
        dialog = ReconstructDestinationDialog(
            session.project_id, plan, self.paths.default_volume_path(session.config), self,
            details=describe_run(session.config, n_rows_total, n_selected),
        )
        if not dialog.exec():
            return ""
        path = dialog.chosen_path()
        if path is not None:
            self.paths.remember_volume_path(path)
        return path

    def _reconstruct_streaming(self, session: ProjectSession, path: str) -> None:
        """Reconstructs the whole volume chunk by chunk into an HDF5
        file (PipelineRun.stream()); the volume is never held in RAM."""
        config, note = streaming_config(session.config, path)
        self.status_message.emit(f"reconstructing all slices of {session.project_id} into {path} ...")
        self.reconstruct_slice_btn.setEnabled(False)
        self.reconstruct_all_btn.setEnabled(False)
        self.busy_changed.emit(True)

        def work() -> str:
            session.run.complete()
            streamer = PipelineRun(config, progress=self._relay)
            streamer.project = session.run.project  # already imported; don't read the file again
            return streamer.stream()

        def done(written_path: str) -> None:
            self.busy_changed.emit(False)
            n_rows_total = DataInterface(session.run.project).shape_of("projections")[1]
            total = len(range(*config.rows.indices(n_rows_total))) if config.rows is not None else n_rows_total
            self.progress_event.emit(ProgressEvent(
                session.project_id, "Done", total, total,
                resolve_device(config.reconstruction_backend, config.device), f"written to {written_path}",
            ))
            self.reconstruct_slice_btn.setEnabled(True)
            self.reconstruct_all_btn.setEnabled(True)
            session.error = None
            session.volume_path = written_path
            session.volume_dataset = config.export_dataset_path
            session.ram_volume = None
            session.exported = True
            message = f"reconstructed {session.project_id} into {written_path}"
            if note:
                message += f" -- {note}"
            self.status_message.emit(message)
            self.session_changed.emit(session)
            self.volume_written.emit(session)

        def failed(exc: Exception) -> None:
            self.busy_changed.emit(False)
            self.reconstruct_slice_btn.setEnabled(True)
            self.reconstruct_all_btn.setEnabled(True)
            session.error = str(exc)
            self.status_message.emit(f"reconstruction failed: {describe_error(exc)}")
            self.progress_event.emit(ProgressEvent(session.project_id, "Failed", message=describe_error(exc)))
            if isinstance(exc, MemoryError):
                QMessageBox.warning(self, "Out of memory", describe_error(exc))

        self._workers.append(run_in_background(work, on_finished=done, on_failed=failed))
