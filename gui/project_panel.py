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

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QMenu,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from orchestration import save_config
from gui.copy_settings_dialog import CopySettingsDialog
from gui.session import ProjectSession
from gui.settings_clipboard import ALL_COPYABLE_FIELDS, copy_settings, paste_settings

INFO_FIELDS = ("Status", "Center of rotation", "Tilt", "Format", "Storage")

# error: import itself never produced a project (no plugin matched, or
# some other exception) -- not fixable from this panel, a genuinely
# different kind of broken than the other three.
# needs_input: imported, but a mandatory_soft field (e.g. geometry_type)
# is still unresolved -- fixable via Basic settings.
# ready / reconstructed: the original 3-state plan (see
# doc/GUI_ARCHITECTURE.md); a 4th color was added for "error" rather
# than reusing "needs_input"'s, since only one of the two is something
# filling in a field can fix.
_STATUS_COLORS = {
    "importing": QColor("#2980b9"),
    "error": QColor("#c0392b"),
    "needs_input": QColor("#d68910"),
    "ready": QColor("#7f8c8d"),
    "reconstructed": QColor("#27ae60"),
}


def _status_of(session: ProjectSession) -> tuple[str, str]:
    """(state, label) -- state keys into _STATUS_COLORS; label is what's
    shown as this row's Status value."""
    if session.run is None:
        if session.error is not None:
            return "error", f"error: {session.error}"
        return "importing", "importing..."
    if session.run.project.needs_user_input():
        missing = ", ".join(session.run.project.missing_soft_fields())
        return "needs_input", f"needs input: {missing}"
    if session.exported or session.run.volume is not None:
        return "reconstructed", "reconstructed"
    return "ready", "ready"


def _info_values(session: ProjectSession) -> tuple[str, str, str, str, str]:
    _, status_label = _status_of(session)
    config = session.config
    center = f"{config.center:.2f}" if config.center is not None else "auto"
    tilt = f"{config.tilt_deg:.3f}" if config.tilt_deg is not None else "auto"
    return status_label, center, tilt, config.format, config.storage_mode


class ProjectPanel(QWidget):
    project_opened = Signal(object)  # ProjectSession
    reconstruct_selected_requested = Signal()  # checked_sessions(), see MainWindow._reconstruct_selected()
    sessions_removed = Signal(list)  # list[ProjectSession] just deleted, for MainWindow to clear if one was open
    settings_pasted = Signal(list)  # list[ProjectSession] just pasted onto, for MainWindow to refresh if one was open
    config_exported = Signal(list)  # list[ProjectSession] whose config .ini was just written

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sessions: list[ProjectSession] = []
        self._items: dict[int, QTreeWidgetItem] = {}  # id(session) -> its top-level item
        self._clipboard: dict | None = None  # last-copied settings, see gui/settings_clipboard.py

        self.tree = QTreeWidget(self)
        self.tree.setHeaderLabels(["Project", "Value"])
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.tree.customContextMenuRequested.connect(self._on_context_menu_requested)

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
            child = QTreeWidgetItem([label, value])
            child.setFlags(child.flags() & ~Qt.ItemIsSelectable)  # only project rows are selectable
            item.addChild(child)

        self.tree.addTopLevelItem(item)
        self.sessions.append(session)
        self._items[id(session)] = item
        self._apply_status_color(session, item)

    def remove_session(self, session: ProjectSession) -> None:
        item = self._items.pop(id(session), None)
        if item is None:
            return
        index = self.tree.indexOfTopLevelItem(item)
        if index != -1:
            self.tree.takeTopLevelItem(index)
        if session in self.sessions:
            self.sessions.remove(session)

    def refresh(self, session: ProjectSession) -> None:
        item = self._items.get(id(session))
        if item is None:
            return
        item.setText(0, session.project_id)
        values = _info_values(session)
        for i, value in enumerate(values):
            child = item.child(i)
            child.setText(1, value)
            child.setToolTip(1, value)
        self._apply_status_color(session, item)

    def _apply_status_color(self, session: ProjectSession, item: QTreeWidgetItem) -> None:
        state, _ = _status_of(session)
        color = _STATUS_COLORS[state]
        item.setForeground(0, color)
        item.child(0).setForeground(1, color)  # the Status child row itself

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

    # -- right-click menu: delete / copy / paste settings --------------------

    def _on_context_menu_requested(self, pos) -> None:
        item = self.tree.itemAt(pos)
        if item is None:
            return
        top_level = item.parent() or item
        if top_level not in self.tree.selectedItems():
            self.tree.setCurrentItem(top_level)  # right-clicking outside the current selection replaces it

        sessions = [it.data(0, Qt.UserRole) for it in self.tree.selectedItems() if it.parent() is None]
        sessions = [s for s in sessions if s is not None]
        if not sessions:
            return

        menu = self._build_context_menu(sessions)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _build_context_menu(self, sessions: list[ProjectSession]) -> QMenu:
        menu = QMenu(self)
        if len(sessions) == 1:
            # Copying more than one project's settings at once has no
            # single meaning, so these two are hidden entirely for a
            # multi-selection rather than acting on just one of them.
            session = sessions[0]
            menu.addAction("Copy settings", lambda: self._copy_settings(session, ALL_COPYABLE_FIELDS))
            menu.addAction("Copy selected settings...", lambda: self._copy_selected_settings(session))

        paste_action = menu.addAction("Paste settings", lambda: self._paste_settings(sessions))
        paste_action.setEnabled(self._clipboard is not None)

        menu.addSeparator()
        config_plural = "s" if len(sessions) > 1 else ""
        menu.addAction(f"Export config{config_plural}", lambda: self._export_configs(sessions))
        if len(sessions) == 1:
            # one chosen path can't serve more than one project, same
            # reasoning as Copy settings/Copy selected settings above.
            menu.addAction("Export config to...", lambda: self._export_config_as(sessions[0]))

        menu.addSeparator()
        plural = "s" if len(sessions) > 1 else ""
        menu.addAction(f"Delete project{plural}", lambda: self._delete_sessions(sessions))
        return menu

    def _copy_settings(self, session: ProjectSession, field_names: list[str]) -> None:
        self._clipboard = copy_settings(session, field_names)

    def _copy_selected_settings(self, session: ProjectSession) -> None:
        dialog = CopySettingsDialog(self)
        if dialog.exec():
            self._copy_settings(session, dialog.selected_fields())

    def _paste_settings(self, sessions: list[ProjectSession]) -> None:
        if self._clipboard is None:
            return
        for session in sessions:
            paste_settings(session, self._clipboard)
            self.refresh(session)
        self.settings_pasted.emit(sessions)

    def _delete_sessions(self, sessions: list[ProjectSession]) -> None:
        plural = "s" if len(sessions) > 1 else ""
        names = ", ".join(s.project_id for s in sessions)
        confirmed = QMessageBox.question(
            self, f"Delete project{plural}",
            f"Close and free the memory for project{plural} {names}? This can't be undone -- "
            "re-import the file(s) to bring them back.",
        )
        if confirmed != QMessageBox.Yes:
            return
        for session in sessions:
            self.remove_session(session)
        self.sessions_removed.emit(sessions)

    def _default_config_path(self, session: ProjectSession) -> Path:
        return Path(session.config.path).parent / f"{session.project_id}_config.ini"

    def _export_configs(self, sessions: list[ProjectSession]) -> None:
        for session in sessions:
            save_config(session.config, self._default_config_path(session))
        self.config_exported.emit(sessions)

    def _export_config_as(self, session: ProjectSession) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save config file", str(self._default_config_path(session)), "Config files (*.ini)",
        )
        if not path:
            return
        save_config(session.config, path)
        self.config_exported.emit([session])
