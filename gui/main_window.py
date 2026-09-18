"""
MainWindow: the project panel, viewer, and adjustment panel as
dockable widgets around a menu bar and status bar. See
doc/GUI_ARCHITECTURE.md.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QMainWindow, QProgressBar

from core.data_interface import DataInterface
from data_import.errors import AmbiguousVendorMatchError, NoVendorMatchError
from orchestration import BatchReport, PipelineRun, PipelineStageError, ProjectConfig, run_batch

from gui import ram_check
from gui.adjustment_panel import AdjustmentPanel
from gui.import_dialog import ImportDialog
from gui.plugin_picker_dialog import PluginPickerDialog
from gui.project_panel import ProjectPanel
from gui.session import ProjectSession, unique_project_id
from gui.viewer import ImageViewer
from gui.workers import run_in_background


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("tomo_project")
        self.resize(1400, 900)
        self._workers: list = []
        self._busy_count = 0

        self.viewer = ImageViewer(self)
        self.setCentralWidget(self.viewer)

        self.project_panel = ProjectPanel(self)
        self.adjustment_panel = AdjustmentPanel(self.viewer, self)

        # Not closable -- there's no way back in once closed (no View
        # menu/toolbar toggle exists), so a user who closes one by
        # accident would be stuck. Still movable/floatable.
        _not_closable = QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable

        left_dock = QDockWidget("Projects", self)
        left_dock.setWidget(self.project_panel)
        left_dock.setFeatures(_not_closable)
        self.addDockWidget(Qt.LeftDockWidgetArea, left_dock)

        right_dock = QDockWidget("Adjustment", self)
        right_dock.setWidget(self.adjustment_panel)
        right_dock.setFeatures(_not_closable)
        self.addDockWidget(Qt.RightDockWidgetArea, right_dock)

        self.project_panel.project_opened.connect(self._open_project)
        self.project_panel.reconstruct_selected_requested.connect(self._reconstruct_selected)
        self.adjustment_panel.status_message.connect(self._show_status)
        self.adjustment_panel.busy_changed.connect(self._on_busy_changed)
        self.adjustment_panel.session_changed.connect(self.project_panel.refresh)

        # Indeterminate (no known percentage) -- every pipeline call is
        # start/finish only, not real progress. Visible only while at
        # least one background operation (import, reconstruct, find
        # COR/tilt) is in flight; see _push_busy()/_pop_busy().
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setMaximumWidth(150)
        self._progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self._progress_bar)

        self._build_menu()
        self.statusBar().showMessage("Ready")

    # -- busy indicator -----------------------------------------------------

    def _on_busy_changed(self, busy: bool) -> None:
        self._push_busy() if busy else self._pop_busy()

    def _push_busy(self) -> None:
        self._busy_count += 1
        self._progress_bar.setVisible(True)

    def _pop_busy(self) -> None:
        self._busy_count = max(0, self._busy_count - 1)
        if self._busy_count == 0:
            self._progress_bar.setVisible(False)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction("Import...", lambda: self._import(storage_mode="disk"))
        file_menu.addAction("Import to RAM...", lambda: self._import(storage_mode="ram"))

        # Standalone tools -- tiff/hdf5 conversion, a plugin wizard, etc.
        # Each opens its own window; none of it shares state with the
        # main app beyond the files it's given. Not designed yet.
        tools_menu = self.menuBar().addMenu("Tools")
        tools_menu.setEnabled(False)

    # -- import ---------------------------------------------------------------

    def _import(self, storage_mode: str) -> None:
        dialog = ImportDialog(storage_mode, self)
        if not dialog.exec():
            return

        existing_ids = {s.project_id for s in self.project_panel.sessions}
        for path, format in dialog.staged_items():
            project_id = unique_project_id(Path(path).stem, existing_ids)
            existing_ids.add(project_id)
            config = ProjectConfig(
                project_id=project_id,
                path=path,
                format=format,
                storage_mode=storage_mode,
                # None means "auto-find" to PipelineRun.calibrate(); the
                # GUI never wants that implicitly, only via Automatic
                # adjustments' find COR/find tilt buttons. `center` stays
                # None as a GUI-only sentinel until AdjustmentPanel knows
                # this project's width and can resolve it to the image's
                # midpoint (not an arbitrary absolute column); tilt has
                # no such dependency, 0 degrees is a sensible guess for
                # any width.
                center=None,
                tilt_deg=0.0,
                # Next to the source file, not PipelineRun's own default
                # (the current working directory) -- an export a user
                # didn't explicitly place somewhere else should end up
                # next to the data it came from.
                export_path=str(Path(path).parent / f"{project_id}_volume.h5"),
            )
            session = ProjectSession(config=config)
            self.project_panel.add_session(session)
            self._import_session(session)

    def _import_session(self, session: ProjectSession) -> None:
        def work() -> PipelineRun:
            run = PipelineRun(session.config)
            run.import_data()
            try:
                run.complete()
            except PipelineStageError as e:
                session.error = str(e)
            return run

        def done(run: PipelineRun) -> None:
            self._pop_busy()
            session.run = run
            self.project_panel.refresh(session)
            self._show_status(f"imported {session.project_id}")

        def failed(exc: Exception) -> None:
            self._pop_busy()
            session.error = str(exc)
            self.project_panel.refresh(session)
            self._show_status(f"import failed for {session.project_id}: {exc}")
            # session.config.plugin_name already set means the user
            # already picked once for this session and it still failed
            # -- offer again (with Cancel), regardless of the new
            # exception type, rather than only on the original
            # detection failure.
            if session.config.plugin_name is not None or isinstance(exc.__cause__, (NoVendorMatchError, AmbiguousVendorMatchError)):
                self._offer_plugin_picker(session)

        self._push_busy()
        self._workers.append(run_in_background(work, on_finished=done, on_failed=failed))

    def _offer_plugin_picker(self, session: ProjectSession) -> None:
        dialog = PluginPickerDialog(session.config.format, session.project_id, session.error or "", self)
        if not dialog.exec():
            # Cancel -- no plugin is ever going to open this file. It was
            # only ever in the list provisionally, as an "importing"
            # placeholder; giving up removes it rather than leaving a
            # permanently-broken row behind.
            self._show_status(f"removed {session.project_id}: no plugin available")
            self.project_panel.remove_session(session)
            return
        session.config.plugin_name = dialog.selected_plugin_name()
        session.error = None
        self.project_panel.refresh(session)
        self._import_session(session)  # retry; failed() loops back here again if this plugin doesn't work either

    # -- reconstructing every checked project ------------------------------------

    def _reconstruct_selected(self) -> None:
        """
        "Run" from the three-tier reconstruct design: unlike
        AdjustmentPanel's two reconstruct actions (one open project,
        preview only), this exports, via orchestration.run_batch(),
        the same function `tomo batch` uses -- sequentially, one
        project's failure never stopping the rest.
        """
        checked = self.project_panel.checked_sessions()
        if not checked:
            self._show_status("no projects checked")
            return

        configs, skipped = [], []
        for session in checked:
            if not session.imported:
                skipped.append(f"{session.project_id}: still importing")
                continue
            _, n_rows, width = DataInterface(session.run.project).shape_of("projections")

            # Save settings' from/to slice range and crop box are GUI-only
            # (ProjectSession.export_rows/export_crop_yx) until this exact
            # moment -- translated into config.rows/config.crop only here,
            # right before the one action that actually exports, so the
            # RAM estimate below reflects the real output size rather than
            # the full volume's.
            if session.export_rows is not None and session.export_rows != (0, n_rows - 1):
                start, end = session.export_rows
                start, end = max(0, start), min(n_rows - 1, end)
                session.config.rows = slice(start, end + 1)
                effective_n_rows = max(end - start + 1, 0)
            else:
                session.config.rows = None
                effective_n_rows = n_rows
            session.config.crop = (
                ((0, effective_n_rows), session.export_crop_yx[0:2], session.export_crop_yx[2:4])
                if session.export_crop_yx is not None
                else None
            )

            required = ram_check.estimate_volume_bytes(effective_n_rows, width)
            if not ram_check.fits_in_available_ram(required):
                skipped.append(f"{session.project_id}: needs roughly {ram_check.describe(required)}, not enough RAM available")
                continue
            configs.append(session.config)

        if skipped:
            self._show_status("skipped: " + "; ".join(skipped))
        if not configs:
            return

        sessions_by_id = {s.project_id: s for s in checked}

        def done(report: BatchReport) -> None:
            self._pop_busy()
            succeeded = [r.project_id for r in report.succeeded]
            failed_results = [(r.project_id, r.error) for r in report.failed]
            for project_id in succeeded:
                session = sessions_by_id.get(project_id)
                if session is not None:
                    session.exported = True
                    self.project_panel.refresh(session)
            message = f"reconstruct selected: {len(succeeded)} succeeded"
            if failed_results:
                message += f", {len(failed_results)} failed (" + "; ".join(f"{pid}: {err}" for pid, err in failed_results) + ")"
            self._show_status(message)

        def failed(exc: Exception) -> None:
            self._pop_busy()
            self._show_status(f"reconstruct selected failed: {exc}")

        self._show_status(f"reconstructing {len(configs)} selected project(s) ...")
        self._push_busy()
        self._workers.append(run_in_background(run_batch, configs, on_finished=done, on_failed=failed))

    # -- opening a project ------------------------------------------------------

    def _open_project(self, session: ProjectSession) -> None:
        if not session.imported:
            if session.error is not None:
                self._show_status(f"{session.project_id} failed to import: {session.error}")
            else:
                self._show_status(f"{session.project_id} is still importing")
            return
        di = DataInterface(session.run.project)
        n_angles = di.shape_of("projections")[0]
        # get_projections(i) reads just that one frame regardless of
        # format/storage_mode (one h5py slice, or one tiff file) --
        # get_projections() with no index would read the entire stack
        # just to open the project, defeating storage_mode="disk"
        # entirely.
        self.viewer.show_stack_lazy(di.get_projections, n_angles)
        self.adjustment_panel.set_session(session)
        self._show_status(f"opened {session.project_id}")

    def _show_status(self, message: str) -> None:
        self.statusBar().showMessage(message)
