"""
PluginPickerDialog: shown when a project's vendor plugin couldn't be
detected automatically (no match, or an ambiguous one), letting the
user force a specific registered plugin. See
MainWindow._offer_plugin_picker().
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QLabel, QVBoxLayout

from data_import.plugin import get_plugins


class PluginPickerDialog(QDialog):
    def __init__(self, format: str, project_id: str, error_message: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Select a plugin for '{project_id}'")
        self.resize(480, 220)

        label = QLabel(f"Could not detect a plugin for '{project_id}' ({format}):\n\n{error_message}", self)
        label.setWordWrap(True)

        self.combo = QComboBox(self)
        names = [p.name for p in get_plugins(format)]
        self.combo.addItems(names)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Ok).setEnabled(bool(names))

        layout = QVBoxLayout(self)
        layout.addWidget(label)
        layout.addWidget(self.combo)
        layout.addWidget(buttons)

    def selected_plugin_name(self) -> Optional[str]:
        return self.combo.currentText() or None
