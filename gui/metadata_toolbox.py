"""
The metadata toolbox: every metadata field of the selected project, grouped by
tier, with its value, where it came from, and what is missing. Shown as a dock
in the main window (Tools > Metadata toolbox).
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QCheckBox, QLabel, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from gui.metadata_info import GROUPS, STATUS_TEXT, MetadataView, metadata_view
from gui.session import ProjectSession

_STATUS_COLORS = {
    "missing": QColor("#c0392b"),
    "later": QColor("#7f8c8d"),
    "not_needed": QColor("#7f8c8d"),
    "default": QColor("#b9770e"),
    "derived": QColor("#1f618d"),
    "user": QColor("#1e8449"),
    "settings": QColor("#1e8449"),
}


class MetadataToolbox(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.session: Optional[ProjectSession] = None
        self.view: MetadataView = metadata_view(None)

        self.title_label = QLabel(self)
        self.title_label.setStyleSheet("font-weight: bold;")
        self.summary_label = QLabel(self)
        self.summary_label.setWordWrap(True)
        self.only_problems = QCheckBox("Only what is missing or not from the file", self)
        self.only_problems.toggled.connect(lambda _checked: self._fill())

        self.tree = QTreeWidget(self)
        self.tree.setHeaderLabels(["Field", "Value", "Status", "Unit"])
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.title_label)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.only_problems)
        layout.addWidget(self.tree)
        self._fill()

    def show_session(self, session: Optional[ProjectSession]) -> None:
        self.session = session
        self.refresh()

    def refresh_if(self, session: ProjectSession) -> None:
        """Called whenever a session changed; re-reads it only if it is the one shown."""
        if session is self.session:
            self.refresh()

    def refresh(self) -> None:
        self.view = metadata_view(self.session)
        self._fill()

    def _fill(self) -> None:
        view = self.view
        self.title_label.setText(view.title or "No project selected")
        self.summary_label.setText(view.summary)
        self.tree.clear()
        only_problems = self.only_problems.isChecked()
        for name, label in GROUPS:
            rows = view.group(name)
            if only_problems:
                rows = [r for r in rows if r.status != "read"]
            if not rows:
                continue
            missing = sum(1 for r in rows if r.status == "missing")
            head = QTreeWidgetItem([f"{label}  ({len(rows)}{f', {missing} missing' if missing else ''})", "", "", ""])
            font = head.font(0)
            font.setBold(True)
            head.setFont(0, font)
            self.tree.addTopLevelItem(head)
            for row in rows:
                item = QTreeWidgetItem([row.key, row.value, STATUS_TEXT.get(row.status, row.status), row.unit])
                item.setToolTip(0, row.note or row.key)
                item.setToolTip(1, row.value)
                if row.source:
                    item.setToolTip(2, f"source: {row.source}")
                color = _STATUS_COLORS.get(row.status)
                if color is not None:
                    item.setForeground(2, QBrush(color))
                    if row.status == "missing":
                        item.setForeground(0, QBrush(color))
                head.addChild(item)
            head.setExpanded(True)
        for column in range(3):
            self.tree.resizeColumnToContents(column)
