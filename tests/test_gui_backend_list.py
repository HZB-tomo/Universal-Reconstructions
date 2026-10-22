"""The GUI's reconstruction-algorithm list comes from the backend registry."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6.QtWidgets import QApplication  # noqa: E402

from reconstruction import list_backends  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_algorithm_combo_lists_exactly_the_usable_backends_in_order(app):
    from gui.adjustment_panel import BasicSettingsTab

    tab = BasicSettingsTab()
    items = [tab.backend_combo.itemText(i) for i in range(tab.backend_combo.count())]
    assert items == list_backends()
    assert items[0] == "algotom_fbp"                       # the default stays first


def test_a_backend_registered_later_appears_in_new_panels(app, monkeypatch):
    import reconstruction.backend as backend_module
    from gui.adjustment_panel import BasicSettingsTab
    from reconstruction.backend import ReconstructionBackend

    class Custom(ReconstructionBackend):
        name = "custom_for_gui_test"
        supported_geometries = {"parallel"}

        def reconstruct(self, *a, **k):
            raise NotImplementedError

    monkeypatch.setattr(backend_module, "_REGISTRY", {**backend_module._REGISTRY, "custom_for_gui_test": Custom()})
    tab = BasicSettingsTab()
    assert "custom_for_gui_test" in [tab.backend_combo.itemText(i) for i in range(tab.backend_combo.count())]
