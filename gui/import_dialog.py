"""
ImportDialog: stage hdf5 files and tiff project folders, then import
them all at once. File > Import and File > Import to RAM both open
this, differing only in the storage_mode the caller passes in.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTreeView,
    QVBoxLayout,
)


def select_multiple_directories(parent, title: str) -> list[str]:
    # QFileDialog.getExistingDirectory only ever returns one path;
    # multi-directory selection needs the non-native dialog with
    # its internal views switched to multi-selection.
    dialog = QFileDialog(parent, title)
    dialog.setFileMode(QFileDialog.Directory)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    list_view = dialog.findChild(QListView, "listView")
    if list_view is not None:
        list_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
    tree_view = dialog.findChild(QTreeView)
    if tree_view is not None:
        tree_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
    if dialog.exec():
        return dialog.selectedFiles()
    return []


class ImportDialog(QDialog):
    def __init__(self, storage_mode: str, parent=None):
        super().__init__(parent)
        self.storage_mode = storage_mode
        self.setWindowTitle(f"Import ({storage_mode})")
        self.resize(520, 420)

        self.list_widget = QListWidget(self)
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)

        open_file_btn = QPushButton("Open File", self)
        open_file_btn.clicked.connect(self._open_files)
        open_folder_btn = QPushButton("Open Folder(s)", self)
        open_folder_btn.clicked.connect(self._open_folders)
        remove_btn = QPushButton("Remove", self)
        remove_btn.clicked.connect(self._remove_selected)

        button_row = QHBoxLayout()
        button_row.addWidget(open_file_btn)
        button_row.addWidget(open_folder_btn)
        button_row.addStretch()
        button_row.addWidget(remove_btn)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel, self)
        import_btn = buttons.addButton("Import", QDialogButtonBox.AcceptRole)
        import_btn.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(button_row)
        layout.addWidget(self.list_widget)
        layout.addWidget(buttons)

    def _open_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open HDF5 file(s)", "", "HDF5 files (*.h5 *.hdf5)"
        )
        for path in paths:
            self._stage(path, "hdf5")

    def _open_folders(self) -> None:
        for path in self._select_multiple_directories():
            self._stage(path, "tiff")

    def _select_multiple_directories(self) -> list[str]:
        return select_multiple_directories(self, "Open TIFF project folder(s)")

    def _stage(self, path: str, format: str) -> None:
        already_staged = {self.list_widget.item(i).data(Qt.UserRole)[0] for i in range(self.list_widget.count())}
        if path in already_staged:
            return
        item = QListWidgetItem(f"[{format}] {path}")
        item.setData(Qt.UserRole, (path, format))
        self.list_widget.addItem(item)

    def _remove_selected(self) -> None:
        for item in self.list_widget.selectedItems():
            self.list_widget.takeItem(self.list_widget.row(item))

    def staged_items(self) -> list[tuple[str, str]]:
        return [self.list_widget.item(i).data(Qt.UserRole) for i in range(self.list_widget.count())]
