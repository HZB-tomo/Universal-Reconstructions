"""gui/convert_dialog.py: drives the dialog like a user, offscreen."""
import os
import time

import h5py
import numpy as np
import pytest
import tifffile

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.convert_dialog import ConvertDialog, output_path_for  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _project(root, name, n=6):
    folder = root / name
    (folder / "Proj").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        tifffile.imwrite(folder / "Proj" / f"proj_{i:04d}.tif", rng.integers(0, 4000, (8, 12), dtype=np.uint16))
    tifffile.imwrite(folder / "FlatsNeutrons.tif", np.full((8, 12), 3000, np.uint16))
    tifffile.imwrite(folder / "DarksNeutrons.tif", np.full((8, 12), 50, np.uint16))
    (folder / "scan_information.xml").write_text('<scanInformation><imager pixelSize_mm="0.05"/></scanInformation>')
    return folder


def _wait(app, dialog, timeout=20):
    deadline = time.time() + timeout
    while dialog._running and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert not dialog._running, "conversion did not finish"


def test_output_path_naming(tmp_path):
    assert output_path_for(str(tmp_path / "scan"), "", 1) == tmp_path / "scan.h5"
    assert output_path_for(str(tmp_path / "scan"), "", 4) == tmp_path / "scan_bin4.h5"
    assert output_path_for(str(tmp_path / "scan"), str(tmp_path / "out"), 2) == tmp_path / "out" / "scan_bin2.h5"


def test_dialog_converts_several_projects_with_binning(app, tmp_path):
    a, b = _project(tmp_path, "scan_a"), _project(tmp_path, "scan_b")
    out = tmp_path / "out"
    out.mkdir()
    dialog = ConvertDialog()
    dialog.sources.addItem(str(a))
    dialog.sources.addItem(str(b))
    dialog.output_dir.setText(str(out))
    dialog.bin_spin.setValue(2)

    dialog.convert_btn.click()
    assert dialog._running and not dialog.convert_btn.isEnabled() and dialog.stop_btn.isEnabled()
    _wait(app, dialog)

    for name in ("scan_a_bin2.h5", "scan_b_bin2.h5"):
        with h5py.File(out / name, "r") as f:
            assert f["/entry/instrument/detector/data"].shape == (6, 4, 6)
            assert f["/entry/instrument/detector/flat_fields"].shape == (1, 4, 6)
            assert f["/entry/instrument/detector/pixel_size"][()] == pytest.approx(100.0)
    assert dialog.log.toPlainText().count(": ok") == 2
    assert dialog.convert_btn.isEnabled() and dialog.close_btn.isEnabled()


def test_dialog_reports_a_failing_project_and_continues(app, tmp_path):
    good = _project(tmp_path, "good")
    empty = tmp_path / "empty"
    empty.mkdir()
    dialog = ConvertDialog()
    dialog.sources.addItem(str(empty))
    dialog.sources.addItem(str(good))

    dialog.convert_btn.click()
    _wait(app, dialog)

    assert (tmp_path / "good.h5").exists()
    assert not (tmp_path / "empty.h5").exists()
    log = dialog.log.toPlainText()
    assert "good.h5: ok" in log and "empty.h5:" in log and "empty.h5: ok" not in log


def test_dialog_needs_a_source_and_cannot_be_closed_while_running(app, tmp_path):
    dialog = ConvertDialog()
    dialog.convert_btn.click()
    assert "Add at least one" in dialog.log.toPlainText() and not dialog._running

    dialog._running = True
    dialog.show()
    dialog.close()
    assert dialog.isVisible()
    dialog._running = False
    dialog.close()


def test_tools_menu_opens_the_dialog(app):
    from gui.main_window import MainWindow

    window = MainWindow()
    tools = next(a for a in window.menuBar().actions() if a.text() == "Tools").menu()
    assert tools.isEnabled()
    action = next(a for a in tools.actions() if "Convert" in a.text())
    action.trigger()
    assert window._convert_dialog is not None and window._convert_dialog.isVisible()
    window._convert_dialog.close()
