"""
ReconstructDestinationDialog: asked before "Reconstruct all slices" --
write the volume to an HDF5 file (streamed, low memory) or keep it in
RAM. See gui/reconstruct_plan.py for the numbers it shows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from gui.reconstruct_plan import ReconstructPlan
from orchestration import ram_check


class ReconstructDestinationDialog(QDialog):
    def __init__(self, project_id: str, plan: ReconstructPlan, default_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reconstruct all slices")
        self._plan = plan

        message = (
            f"You are about to reconstruct the whole volume of {project_id}.\n\n"
            f"Available RAM: {ram_check.describe(plan.available_bytes)}\n"
            f"Needed to reconstruct in memory: about {ram_check.describe(plan.needed_bytes)}\n\n"
            "Do you want to write it as an HDF5 file? The slices are streamed to disk as they are "
            "reconstructed, so memory use stays low, and the file can be reopened any time."
        )
        if not plan.fits_in_ram:
            message += "\n\nThere is not enough RAM to keep the volume in memory: you have to choose a file."

        self.hdf5_radio = QRadioButton("Write to an HDF5 file (streamed)", self)
        self.ram_radio = QRadioButton("Keep in memory only", self)
        self.path_edit = QLineEdit(default_path, self)
        self.browse_btn = QPushButton("Browse...", self)
        self.browse_btn.clicked.connect(self._browse)

        self.hdf5_radio.setEnabled(plan.can_stream)
        self.ram_radio.setEnabled(plan.fits_in_ram)
        if plan.can_stream:
            self.hdf5_radio.setChecked(True)
        else:
            self.ram_radio.setChecked(True)
        self.hdf5_radio.toggled.connect(self._sync_enabled)
        self.path_edit.textChanged.connect(self._sync_enabled)

        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit)
        path_row.addWidget(self.browse_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(message, self))
        layout.addWidget(self.hdf5_radio)
        layout.addLayout(path_row)
        if plan.stream_blocker:
            layout.addWidget(QLabel(f"Not available: {plan.stream_blocker}.", self))
        layout.addWidget(self.ram_radio)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._sync_enabled()

    def _sync_enabled(self, *_args) -> None:
        streaming = self.hdf5_radio.isChecked()
        self.path_edit.setEnabled(streaming)
        self.browse_btn.setEnabled(streaming)
        ok = self.buttons.button(QDialogButtonBox.Ok)
        ok.setEnabled((streaming and bool(self.path_edit.text().strip())) or (not streaming and self._plan.fits_in_ram))

    def _browse(self) -> None:
        # DontConfirmOverwrite: an existing file isn't replaced -- the
        # pipeline writes to a numbered sibling instead (see
        # PipelineRun._resolve_export_path()), so a confirm here would be misleading.
        path, _ = QFileDialog.getSaveFileName(
            self, "Reconstruction output", self.path_edit.text(), "HDF5 files (*.h5 *.hdf5)",
            options=QFileDialog.DontConfirmOverwrite,
        )
        if path:
            self.path_edit.setText(path)

    def chosen_path(self) -> Optional[str]:
        """The HDF5 path, or None if 'Keep in memory only' was chosen."""
        if not self.hdf5_radio.isChecked():
            return None
        path = self.path_edit.text().strip()
        return str(Path(path).with_suffix(".h5")) if Path(path).suffix.lower() not in (".h5", ".hdf5") else path
