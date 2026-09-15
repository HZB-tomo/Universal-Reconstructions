"""
ProjectPanel: every imported project as a checkable tree row, expanded
to show its key settings. Row selection (native, multi) is independent
of the checkbox and of "opened" -- double-click opens a project into
the viewer and adjustment panel without checking or selecting it.

QTreeWidget, not a QAbstractItemModel over QTreeView: item-based
widgets handle checkbox toggling natively, which a raw
Qt.CheckStateRole model turned out not to (see git history --
ProjectPanel was originally a QTableView/QAbstractTableModel).
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from gui.session import ProjectSession

INFO_FIELDS = ("Center of rotation", "Tilt", "Format", "Storage")


def _info_values(session: ProjectSession) -> tuple[str, str, str, str]:
    config = session.config
    center = f"{config.center:.2f}" if config.center is not None else "auto"
    tilt = f"{config.tilt_deg:.3f}" if config.tilt_deg is not None else "auto"
    return center, tilt, config.format, config.storage_mode


class ProjectPanel(QWidget):
    project_opened = Signal(object)  # ProjectSession
    reconstruct_selected_requested = Signal()  # checked_sessions(), see MainWindow._reconstruct_selected()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sessions: list[ProjectSession] = []
        self._items: dict[int, QTreeWidgetItem] = {}  # id(session) -> its top-level item

        self.tree = QTreeWidget(self)
        self.tree.setHeaderLabels(["Project", "Value"])
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)

        self.reconstruct_selected_btn = QPushButton("Reconstruct selected", self)
        self.reconstruct_selected_btn.clicked.connect(self.reconstruct_selected_requested)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tree)
        layout.addWidget(self.reconstruct_selected_btn)

    def add_session(self, session: ProjectSession) -> None:
        item = QTreeWidgetItem([session.project_id, ""])
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setData(0, Qt.UserRole, session)  # before setCheckState, whose itemChanged needs it
        item.setCheckState(0, Qt.Checked if session.checked else Qt.Unchecked)

        for label, value in zip(INFO_FIELDS, _info_values(session)):
            item.addChild(QTreeWidgetItem([label, value]))

        self.tree.addTopLevelItem(item)
        self.sessions.append(session)
        self._items[id(session)] = item

    def refresh(self, session: ProjectSession) -> None:
        item = self._items.get(id(session))
        if item is None:
            return
        item.setText(0, session.project_id)
        for i, value in enumerate(_info_values(session)):
            item.child(i).setText(1, value)

    def checked_sessions(self) -> list[ProjectSession]:
        return [s for s in self.sessions if s.checked]

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if item.parent() is not None or column != 0:
            return  # only a top-level item's checkbox column is checkable
        session = item.data(0, Qt.UserRole)
        if session is not None:
            session.checked = item.checkState(0) == Qt.Checked

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        top_level = item.parent() or item
        session = top_level.data(0, Qt.UserRole)
        if session is not None:
            self.project_opened.emit(session)
