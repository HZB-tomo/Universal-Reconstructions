"""
Entry point for the tomo_project GUI: `python -m gui` or the
`tomo-gui` console script (see pyproject.toml).
"""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.warm_up_backends()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
