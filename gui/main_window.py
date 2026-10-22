"""
MainWindow: the project panel, viewer, and adjustment panel as
dockable widgets around a menu bar and status bar. See
doc/GUI_ARCHITECTURE.md.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QFileDialog, QMainWindow, QProgressBar

from core.data_interface import DataInterface
from data_import.errors import AmbiguousVendorMatchError, NoVendorMatchError
from orchestration import BatchReport, PipelineRun, ProgressEvent, PipelineStageError, ProjectConfig, run_batch

from orchestration import ram_check
from gui.adjustment_panel import AdjustmentPanel
from gui.convert_dialog import ConvertDialog
from gui.import_dialog import ImportDialog
from gui.plugin_picker_dialog import PluginPickerDialog
from gui.project_panel import ProjectPanel
from gui.reconstruct_plan import RememberedPaths, describe_error, save_ram_volume
from gui.session import ProjectSession, unique_project_id
from gui.viewer import ImageViewer
from gui.volume_view import open_volume
from gui.workers import ProgressRelay, run_in_background


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("tomo_project")
        self.resize(1400, 900)
        self._workers: list = []
        self._busy_count = 0
        self._shown = None  # (ProjectSession, "projections" | "reconstruction") the viewer is showing, if any

        self.viewer = ImageViewer(self)
        self.setCentralWidget(self.viewer)

        # Output folders chosen this session, shared so a later action
        # (e.g. one path for all selected projects) can reuse them.
        self.paths = RememberedPaths()

        # Pipeline progress from import and from "Reconstruct selected"; the adjustment panel
        # has its own relay for its reconstructions. All feed the status box.
        self._progress = ProgressRelay(self)
        self._progress.event.connect(self._on_progress)

        self.project_panel = ProjectPanel(self)
        self.adjustment_panel = AdjustmentPanel(self.viewer, self, paths=self.paths)

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

        self.project_panel.import_requested.connect(self._import)
        self.project_panel.project_opened.connect(self._open_project)
        self.project_panel.reconstruct_selected_requested.connect(self._reconstruct_selected)
        self.project_panel.save_requested.connect(self._save_sessions)
        self.project_panel.sessions_removed.connect(self._on_sessions_removed)
        self.project_panel.settings_pasted.connect(self._on_settings_pasted)
        self.project_panel.config_exported.connect(self._on_config_exported)
        self.adjustment_panel.status_message.connect(self._show_status)
        self.adjustment_panel.busy_changed.connect(self._on_busy_changed)
        self.adjustment_panel.session_changed.connect(self.project_panel.refresh)
        self.adjustment_panel.volume_written.connect(self._show_reconstruction)
        self.adjustment_panel.progress_event.connect(self._on_progress)
        self.project_panel.reconstruction_opened.connect(self._show_reconstruction)

        # Indeterminate (no known percentage) -- every pipeline call is
        # start/finish only, not real progress. Visible only while at
        # least one background operation (import, reconstruct, find
        # COR/tilt) is in flight; see _push_busy()/_pop_busy().
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setMaximumWidth(150)
        self._progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self._progress_bar)

        self._convert_dialog = None
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

        # Standalone tools. Each opens its own window; none of it shares
        # state with the main app beyond the files it's given.
        tools_menu = self.menuBar().addMenu("Tools")
        tools_menu.addAction("Convert TIFF project to HDF5...", self._open_convert_dialog)

    def warm_up_backends(self) -> None:
        """Compile the gridrec kernels in the background right after start-up (Numba's cache
        makes later starts quick), so the first preview does not wait for the compiler.
        Called by gui.main; failures are ignored -- the backend just compiles on first use."""
        from reconstruction.backends.gridrec import warm_up

        self._workers.append(run_in_background(warm_up, on_finished=lambda _r: None, on_failed=lambda _e: None))

    def _open_convert_dialog(self) -> None:
        if self._convert_dialog is None:
            self._convert_dialog = ConvertDialog(self)
        self._convert_dialog.show()
        self._convert_dialog.raise_()

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
            run = PipelineRun(session.config, progress=self._progress)
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
            self._on_progress(ProgressEvent(session.project_id, "Done", message="imported"))
            self._show_status(f"imported {session.project_id}")

        def failed(exc: Exception) -> None:
            self._pop_busy()
            session.error = str(exc)
            self._on_progress(ProgressEvent(session.project_id, "Failed", message=str(exc)))
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
            n_angles, n_rows, width = DataInterface(session.run.project).shape_of("projections")

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

            # Streaming (Save settings' "Stream to disk") reconstructs
            # and writes one chunk of rows at a time -- see
            # PipelineRun._run_streaming() -- so the whole-volume RAM
            # estimate below doesn't apply; it manages its own,
            # per-chunk memory footprint instead.
            if not session.config.stream:
                required = ram_check.estimate_inmemory_run_bytes(n_angles, effective_n_rows, width)
                if not ram_check.fits_in_available_ram(required):
                    skipped.append(
                        f"{session.project_id}: needs roughly {ram_check.describe(required)}, not enough RAM "
                        "available -- enable 'Stream to disk' in Save settings to reconstruct it chunk by chunk instead"
                    )
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
            volume_paths = {r.project_id: r for r in report.succeeded}
            for project_id in succeeded:
                session = sessions_by_id.get(project_id)
                if session is not None:
                    session.exported = True
                    result = volume_paths[project_id]
                    if result.config.export_format == "hdf5" and result.volume_path:
                        session.volume_path = result.volume_path  # shows up in the "Reconstructed" row
                        session.volume_dataset = result.config.export_dataset_path
                        session.ram_volume = None
                    self.project_panel.refresh(session)
            message = f"reconstruct selected: {len(succeeded)} succeeded"
            if failed_results:
                message += f", {len(failed_results)} failed (" + "; ".join(f"{pid}: {err}" for pid, err in failed_results) + ")"
            self._on_progress(ProgressEvent(
                "selected projects", "Failed" if failed_results and not succeeded else "Done",
                message=f"{len(succeeded)} succeeded, {len(failed_results)} failed",
            ))
            self._show_status(message)

        def failed(exc: Exception) -> None:
            self._pop_busy()
            self._show_status(f"reconstruct selected failed: {exc}")

        self._show_status(f"reconstructing {len(configs)} selected project(s) ...")
        self._push_busy()
        self._workers.append(run_in_background(run_batch, configs, progress=self._progress, on_finished=done, on_failed=failed))

    # -- saving in-memory reconstructions -----------------------------------------

    def _save_sessions(self, sessions: list) -> None:
        """Writes each session's in-memory reconstruction to an HDF5 file:
        one chosen file for a single project, one file per project in a
        chosen folder for several. Nothing is overwritten (see unique_path())."""
        todo, skipped = [], []
        for session in sessions:
            if session.ram_volume is None:
                skipped.append(f"{session.project_id}: no reconstruction in memory")
            elif session.volume_path is not None:
                skipped.append(f"{session.project_id}: already saved to {session.volume_path}")
            else:
                todo.append(session)
        if not todo:
            self._show_status("nothing to save -- " + ("; ".join(skipped) if skipped else "no projects selected"))
            return

        if len(todo) == 1:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save reconstruction", self.paths.default_volume_path(todo[0].config), "HDF5 files (*.h5 *.hdf5)",
                options=QFileDialog.DontConfirmOverwrite,
            )
            if not path:
                return
            targets = [(todo[0], path)]
            self.paths.remember_volume_path(path)
        else:
            start = self.paths.volume_dir or str(Path(todo[0].config.path).parent)
            folder = QFileDialog.getExistingDirectory(self, "Save reconstructions into folder", start)
            if not folder:
                return
            targets = [(s, str(Path(folder) / f"{s.project_id}_volume.h5")) for s in todo]
            self.paths.remember_volume_path(targets[0][1])

        def work() -> list:
            saved = []
            for session, path in targets:
                settings = {**session.run.project.export_metadata(), **session.run.settings}
                written = save_ram_volume(
                    session.ram_volume, path, settings, session.volume_dataset,
                    progress=self._progress, project_id=session.project_id,
                )
                saved.append((session, written))
            return saved

        def done(saved: list) -> None:
            self._pop_busy()
            for session, written in saved:
                session.volume_path = written
                session.volume_dataset = "/volume"
                self.project_panel.refresh(session)
            self._on_progress(ProgressEvent(saved[-1][0].project_id, "Done", message=f"saved {len(saved)} reconstruction(s)"))
            message = "saved: " + "; ".join(f"{s.project_id} -> {p}" for s, p in saved)
            if skipped:
                message += " | skipped: " + "; ".join(skipped)
            self._show_status(message)

        def failed(exc: Exception) -> None:
            self._pop_busy()
            self._on_progress(ProgressEvent("save", "Failed", message=describe_error(exc)))
            self._show_status(f"save failed: {describe_error(exc)}")

        self._show_status(f"saving {len(targets)} reconstruction(s) ...")
        self._push_busy()
        self._workers.append(run_in_background(work, on_finished=done, on_failed=failed))

    # -- opening a project ------------------------------------------------------

    def _remember_view(self) -> None:
        """Store the viewer's slicing direction / slice / zoom on the
        project it is showing, before something else replaces it."""
        if self._shown is None:
            return
        state = self.viewer.view_state()
        if state is not None:
            session, kind = self._shown
            session.view_states[kind] = state

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
        self._remember_view()
        self.viewer.show_stack_lazy(di.get_projections, n_angles, shape=di.shape_of("projections"), kind="projections")
        self.viewer.restore_view(session.view_states.get("projections"))
        self._shown = (session, "projections")
        self.adjustment_panel.set_session(session)
        self._show_status(f"opened {session.project_id}")

    def _show_reconstruction(self, session: ProjectSession) -> None:
        """Shows the project's full reconstruction (in-memory or from its
        HDF5 file) in the viewer instead of its projections."""
        self._remember_view()
        if session.ram_volume is not None:
            self.viewer.show_stack(session.ram_volume)
        elif session.volume_path is not None:
            try:
                get_section, shape = open_volume(session.volume_path, session.volume_dataset)
            except (OSError, KeyError) as e:
                self._show_status(f"can't open reconstruction of {session.project_id} ({session.volume_path}): {e}")
                return
            self.viewer.show_stack_lazy(get_section, shape[0], shape=shape)
        else:
            self._show_status(f"{session.project_id} has no full reconstruction yet -- use 'Reconstruct all slices'")
            return
        self.viewer.restore_view(session.view_states.get("reconstruction"))
        self._shown = (session, "reconstruction")
        self.adjustment_panel.set_session(session)
        self._show_status(f"showing reconstruction of {session.project_id}")

    def _on_progress(self, event: ProgressEvent) -> None:
        self.project_panel.status_box.on_progress(event)

    def _show_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    # -- project panel: delete / paste settings ------------------------------

    def _on_sessions_removed(self, sessions: list) -> None:
        if self.adjustment_panel.session in sessions:
            self.adjustment_panel.set_session(None)
            self.viewer.clear()
        if self._shown is not None and self._shown[0] in sessions:
            self._shown = None
        self._show_status(f"deleted {len(sessions)} project(s)")

    def _on_settings_pasted(self, sessions: list) -> None:
        # AdjustmentPanel only reflects session.config/export_crop_yx as
        # of the last set_session() call -- re-running it now that
        # ProjectPanel._paste_settings() has changed those values (for
        # whichever pasted-onto session happens to be the open one)
        # brings the visible fields back in sync.
        if self.adjustment_panel.session in sessions:
            self.adjustment_panel.set_session(self.adjustment_panel.session)
        self._show_status(f"pasted settings onto {len(sessions)} project(s)")

    def _on_config_exported(self, sessions: list) -> None:
        self._show_status(f"saved config for {len(sessions)} project(s)")
