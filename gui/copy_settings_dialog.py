"""
CopySettingsDialog: checkbox picker for ProjectPanel's "Copy selected
settings..." action -- which of settings_clipboard.COPYABLE_SETTINGS'
fields to include. One QGroupBox per category (checkable -- toggling
it checks/unchecks every field inside), plus global Select all/
Deselect all. All fields start checked, so accepting without touching
anything copies the same set "Copy settings" (no dialog) would.
"""
from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QGroupBox, QHBoxLayout, QPushButton, QVBoxLayout

from gui.settings_clipboard import COPYABLE_SETTINGS


class CopySettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Copy selected settings")

        self._checkboxes: dict[str, QCheckBox] = {}
        # group -> the field names it holds, so selected_fields() below
        # doesn't need to walk layouts or reverse-search a dict.
        self._group_fields: dict[QGroupBox, list[str]] = {}

        layout = QVBoxLayout(self)

        select_row = QHBoxLayout()
        select_all_btn = QPushButton("Select all", self)
        select_all_btn.clicked.connect(lambda: self._set_all(True))
        deselect_all_btn = QPushButton("Deselect all", self)
        deselect_all_btn.clicked.connect(lambda: self._set_all(False))
        select_row.addWidget(select_all_btn)
        select_row.addWidget(deselect_all_btn)
        select_row.addStretch()
        layout.addLayout(select_row)

        for category, fields in COPYABLE_SETTINGS.items():
            group = QGroupBox(category, self)
            group.setCheckable(True)
            group.setChecked(True)
            group_layout = QVBoxLayout(group)
            for field_name, label in fields:
                checkbox = QCheckBox(label, self)
                checkbox.setChecked(True)
                self._checkboxes[field_name] = checkbox
                group_layout.addWidget(checkbox)
            layout.addWidget(group)
            self._group_fields[group] = [name for name, _ in fields]

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_all(self, checked: bool) -> None:
        for group in self._group_fields:
            group.setChecked(checked)
        for checkbox in self._checkboxes.values():
            checkbox.setChecked(checked)

    def selected_fields(self) -> list[str]:
        """Only fields whose category is checked AND whose own checkbox
        is checked -- unchecking a whole category greys out (but
        doesn't un-tick) its fields, same as any checkable QGroupBox."""
        selected = []
        for group, field_names in self._group_fields.items():
            if not group.isChecked():
                continue
            selected.extend(name for name in field_names if self._checkboxes[name].isChecked())
        return selected
