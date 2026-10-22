"""Automatic-adjustments tab (guess / tolerance / candidates) and the project panel's import shortcuts."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.adjustment_panel import AutomaticAdjustmentsTab  # noqa: E402
from gui.project_panel import ProjectPanel  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_guess_and_tolerance_default_to_auto(app):
    tab = AutomaticAdjustmentsTab()
    assert tab.guess() is None and tab.tolerance() is None
    seen = []
    tab.find_cor_requested.connect(lambda g, t: seen.append((g, t)))
    tab.find_cor_btn.click()
    assert seen == [(None, None)]


def test_a_typed_guess_and_tolerance_are_sent_with_the_search(app):
    tab = AutomaticAdjustmentsTab()
    seen = []
    tab.find_cor_requested.connect(lambda g, t: seen.append((g, t)))
    tab.guess_spin.setValue(120.5)
    tab.tolerance_spin.setValue(15)
    tab.find_cor_btn.click()
    assert seen == [(120.5, 15.0)]
    tab.set_guess(None)
    assert tab.guess() is None


def test_candidates_are_listed_best_first_and_clickable(app):
    tab = AutomaticAdjustmentsTab()
    tab.set_candidates([(147.5, 1.0), (133.25, 0.08), (90.0, float("nan"))], note="searched 64 to 192")
    texts = [tab.candidates_list.item(i).text() for i in range(3)]
    assert texts[0].startswith("147.50") and "1.00" in texts[0]
    assert texts[1].startswith("133.25") and "0.08" in texts[1]
    assert texts[2] == "90.00"  # unknown score: no "match" text
    assert tab.search_note.text() == "searched 64 to 192"
    chosen = []
    tab.candidate_chosen.connect(chosen.append)
    tab.candidates_list.itemClicked.emit(tab.candidates_list.item(1))
    assert chosen == [133.25]
    tab.set_candidates([])
    assert tab.candidates_list.count() == 0


def test_double_click_on_the_empty_project_area_requests_a_disk_import(app):
    panel = ProjectPanel()
    panel.resize(300, 400)
    panel.show()
    requests = []
    panel.import_requested.connect(requests.append)
    QTest.mouseDClick(panel.tree.viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(20, 100))
    assert requests == ["disk"]


def test_right_click_menu_on_the_empty_area_offers_disk_and_ram(app):
    panel = ProjectPanel()
    requests = []
    panel.import_requested.connect(requests.append)
    actions = panel._build_import_menu().actions()
    assert [a.text() for a in actions] == ["Import project...", "Import project to RAM..."]
    for action in actions:
        action.trigger()
    assert requests == ["disk", "ram"]
