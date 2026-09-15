# GUI architecture

The desktop GUI for `tomo_project`, in `gui/`. Built on PySide6 (Qt6,
LGPL) and pyqtgraph for image display. Desktop only, no web interface.

`gui/` depends on `orchestration/` (and, through it, everything else);
nothing outside `gui/` depends back on it.

## Why PySide6 and pyqtgraph

PySide6 over PyQt5/PyQt6: LGPL, no dual-licensing question to track.
Qt6 instead of legacy Qt5.

pyqtgraph over matplotlib for image display: a slice/frame slider,
histogram/levels (contrast) widget, and draggable ROIs (line, box,
rotatable) come built in, which is what browsing projection stacks,
adjusting contrast, and reading a gray value or an angle off the image
needs. matplotlib redraws the whole canvas per update and has none of
this.

## Files

```
gui/
  main.py               entry point (`tomo-gui` / `python -m gui`)
  main_window.py         MainWindow: docks, menu, status bar, busy indicator
  import_dialog.py       ImportDialog
  project_panel.py       ProjectPanel
  adjustment_panel.py     AdjustmentPanel and its four tabs
  viewer.py              ImageViewer
  session.py             ProjectSession, unique_project_id()
  workers.py             Worker, run_in_background()
```

## Window layout

`MainWindow` is a `QMainWindow`. Three `QDockWidget`s (Projects, left;
Adjustment, right), user-arrangeable; `ImageViewer` is the central
widget. Menu bar (`File`, `Tools`) and status bar.

Dock layout is not persisted between launches yet; resets to the
default each time. Persisting it is a `QMainWindow.saveState()`/
`restoreState()` call away, not done.

## ProjectSession

One project as the GUI tracks it: a `ProjectConfig`, the `PipelineRun`
driving it, and GUI-only state (`checked`, `preview_row`, `error`).
`session.config` and `session.run.config` are the same object --
editing a field in the adjustment panel mutates the `ProjectConfig`
`PipelineRun` already holds, no separate sync step.

`ProjectConfig.resolved_project_id()` is just the path stem, so two
files named the same in different folders collide. `session.py`'s
`unique_project_id()` disambiguates against every id already in the
project panel at import time; `MainWindow._import()` is the only
caller.

The GUI never wants `PipelineRun.calibrate()`'s implicit auto-find
(`None` on either field) to run except when the user asks for it via
Automatic adjustments' Find COR/Find tilt buttons. `tilt_deg` is set
to `0.0` at project creation, a reasonable guess regardless of
detector width. `center` starts as `None` -- a GUI-only sentinel,
resolved to the image's midpoint (`width / 2.0`, i.e. "no offset") the
first time `AdjustmentPanel._refresh_from_session()` knows this
project's width, before which it never reaches a pipeline call. An
absolute column like `0` would not be a sensible default; a real
center of rotation sits within about 1-2% of the width from the
midpoint, see below.

## Threading

Every pipeline call (import, calibrate, reconstruct, at any scope) runs
on a `QThreadPool` worker via `workers.run_in_background()`, never on
the GUI thread. `finished`/`failed` are Qt signals, delivered back on
the GUI thread the normal way.

The caller of `run_in_background()` must keep a reference to the
returned `Worker` until it fires (`AdjustmentPanel._workers`,
`MainWindow._workers`) -- nothing else keeps it alive.

`MainWindow` shows a busy indicator (an indeterminate `QProgressBar` in
the status bar) for as long as any background operation is in flight:
`AdjustmentPanel.busy_changed` (emitted around reconstruct/find
COR/find tilt) and `MainWindow`'s own import worker both go through
`_push_busy()`/`_pop_busy()`, a counter rather than a boolean, so
several imports started at once don't hide the bar the moment the
first one finishes.

## Import

`File > Import` and `File > Import to RAM` both open `ImportDialog`,
differing only in the `storage_mode` passed to it. The dialog stages
paths (`Open File`: multi-select hdf5; `Open Folder(s)`: multi-select
tiff project folders) into a list, `Remove` to drop a staged entry,
`Import` commits the whole list. No parent-directory scanning --
selecting a folder is always exactly one tiff project.

On commit, `MainWindow._import()` builds one `ProjectConfig` and
`ProjectSession` per staged item, adds it to the project panel
immediately, and starts `import_data()` + `complete()` for it in the
background (`_import_session()`). A `PipelineStageError` from
`complete()` (e.g. missing `geometry_type`) is recorded on
`session.error`, not raised -- the project still shows up, still
selectable, and the missing field gets a chance to be filled in via
Basic settings before the next reconstruct attempt retries `complete()`.

## Project panel

`ProjectPanel` is a `QTreeWidget`, not a custom `QAbstractItemModel`.
It was originally a `QTableView` over a `QAbstractTableModel` with a
raw `Qt.CheckStateRole` checkbox column; that combination turned out
unreliable for checkbox clicks in practice. `QTreeWidgetItem`'s own
`setCheckState()`/`itemChanged` (the same mechanism BAMline4CT used)
handles it natively, so the panel was rebuilt on that instead.

One top-level item per session (`project_id`, checkable), with four
child rows underneath it: Center of rotation, Tilt, Format, Storage
(`INFO_FIELDS`/`_info_values()`). `refresh(session)` updates an
existing item's text in place (`ProjectPanel._items`, keyed by
`id(session)`); the tree is never torn down and rebuilt.
`AdjustmentPanel.session_changed` (emitted whenever center/tilt
actually change, manually or via Find COR/Find tilt) is what
`MainWindow` connects to `refresh()`.

Three independent things a row can be:

- **checked**: the top-level item's checkbox. Marks membership in a
  future batch action (`checked_sessions()`). Not implemented yet.
- **selected**: `QTreeWidget`'s native row selection.
- **opened**: `itemDoubleClicked` on the top-level item or any of its
  children (`item.parent() or item` resolves either to the session),
  emits `project_opened`. `MainWindow._open_project()` loads that
  project's projections into the viewer and points the adjustment
  panel at it. Independent of checked/selected.

No status color (red/gray/green) yet. Planned mapping, for when it's
built:

| Color | Condition |
|---|---|
| red | `project.needs_user_input()` |
| gray | resolved, not yet reconstructed |
| green | `run.volume` is not `None` |

## Adjustment panel

`AdjustmentPanel` holds a `QTabWidget` (Basic settings, Automatic
adjustments, Refinements, Save settings) plus two buttons below it,
outside the tabs: Reconstruct (slice) and Reconstruct all slices.
Placement outside the tabs is deliberate -- both always act on
`self.session`, the one currently open project, regardless of which
tab is active.

`set_session(session)` is the only entry point that changes which
project the panel edits; it disables the whole panel when `session` is
`None`.

### Basic settings

Geometry type, slice (z, `session.preview_row`, not a `ProjectConfig`
field), reconstruction algorithm, center of rotation, tilt, and a
"Load config file..." button (`orchestration.load_config()` on a
single-project `.ini`, replacing `session.config` -- keeping
`project_id` -- and resetting `session.run` to `None` so the next
action starts a fresh `PipelineRun` against the new config).

`preview_row` starts as `None` and resolves to the middle row the
first time `_refresh_from_session()` knows this project's row count,
the same pattern `center` uses for `width / 2.0` -- an arbitrary `0`
is not a useful default for either.

Completion's "ask" fields (currently just `geometry_type`) live here,
not in a separate dialog: changing the combo box writes directly into
`session.config.completion_answers`.

### Automatic adjustments

Find center of rotation: `build_reference()` +
`normalize_slice()` on the stack's middle row + `find_center()`.
Writes the result into `session.config.center` and the Basic settings
field.

Find tilt: forces `session.config.rows = None`, calls
`PipelineRun.preprocess()` (full stack), then `find_tilt()` on
`run.normalized`. Writes `tilt_deg` the same way.

Both flag the result in the status message, not block it, if it falls
outside what a real setup produces: `center` more than
`MAX_PLAUSIBLE_COR_OFFSET_FRACTION` (2%) of the width from the image's
midpoint, or `abs(tilt_deg)` over `MAX_PLAUSIBLE_TILT_DEG` (1 degree).
Both constants live at the top of `adjustment_panel.py`. `center` is
an absolute column index end to end (`calibration.find_center()`
through `ProjectConfig.center` through `reconstruction.reconstruct()`,
confirmed against `calibration/backends/algotom_cor.py` and
`reconstruction/backends/algotom_fbp.py`), not a deviation from
center, despite `core/known_fields.py`'s `center_of_rotation` unit
string saying otherwise -- that appears to be a stale doc comment
elsewhere, unrelated to `gui/`.

Each `done()` handler also calls `self._reconstruct(all_slices=False)`
once it's written the new value, so the viewer's preview reflects the
correction immediately rather than waiting for a separate click.

### Refinements

Placeholder. Ring/zinger removal (`ProjectConfig.zinger_removal`,
`ring_correction`, ...) is not wired to any control yet.

### Save settings

`dtype` and `export_format` only. Crop, clip, and rotation are not
wired up: crop and clip need viewer-driven controls (a `RectROI` on a
reconstructed slice), not typed fields, and there is no backend
concept of a post-reconstruction rotation yet -- `tilt_deg` is
rotation-axis tilt, corrected during calibration, not an output-image
rotation.

Nothing in this tab does anything yet; there is no `Run`/export action
in this iteration.

### Reconstruct scope

Three actions, distinguished by which panel owns the button rather
than by label, so scope is never ambiguous:

| Action | Panel | Scope | Rows |
|---|---|---|---|
| Reconstruct (slice) | Adjustment | one project, current row | `slice(preview_row, preview_row + 1)` |
| Reconstruct all slices | Adjustment | one project, every row | `None` |
| Reconstruct selected | Project | every checked project | per project, exports |

Both adjustment-panel actions call
`_run_pipeline_to_reconstruction()`: `import_data()` if not already
done, `complete()`, `preprocess()`, `calibrate()`, `reconstruct()`.
Neither exports; both only ever update `self.viewer`.

`ProjectPanel.reconstruct_selected_btn` ("Reconstruct selected") is the
only action that exports. `MainWindow._reconstruct_selected()` takes
`checked_sessions()`'s configs and passes them straight to
`orchestration.run_batch()` -- the same function `tomo batch` uses --
in one background worker, so the batch stays sequential (`run_batch()`
is deliberately sequential itself; see ORCHESTRATION.md) and one
project's failure doesn't stop the rest. Each `PipelineConfig`'s
`export_path` is set at import time (`MainWindow._import()`) to
`<source file's directory>/<project_id>_volume.h5`, next to the data
it came from, rather than `PipelineRun`'s own default of the current
working directory.

A project still importing, or a project whose reconstructed volume is
estimated not to fit in available RAM (see `gui/ram_check.py`), is
skipped before the batch starts, reported in the status bar, and not
passed to `run_batch()`.

## RAM check

`reconstruct()`/`postprocess()`/`export()` all take and return a full
in-memory array; nothing streams a reconstruction to disk slice by
slice, so the whole output volume has to fit in RAM regardless of
whether it's headed for the viewer (Reconstruct all slices) or a file
(Reconstruct selected). `gui/ram_check.py`'s
`estimate_volume_bytes(n_rows, width)` assumes a square `(width,
width)` output slice (true for `algotom_fbp`, the only registered
backend) at float32, times `SAFETY_FACTOR` (2.5, a margin for backend
buffers and a dtype-conversion in `postprocess()` briefly holding both
the old and new array, not a measured figure) against
`psutil.virtual_memory().available`. Under that, the reconstruction is
refused before it starts rather than left to hang or thrash.

Streaming a reconstruction to disk without holding the full volume in
RAM would remove the need for this check for large projects, but isn't
built (would need `reconstruction/`'s API to change, not just `gui/`).

## Viewer

`ImageViewer` wraps a single `pyqtgraph.ImageView`, but hides its
built-in frame slider (`ImageView.ui.roiPlot`) and drives its own
(`_frame_slider`/`_frame_label`) instead, through one shared path
(`_set_source()`/`_show_frame()`/`_frame_at()`) for both of its two
sources:

- `show_stack(array)` / `show_image(array)`: an eager, fully-in-memory
  numpy array -- a reconstructed volume/slice. `reconstruct()` always
  returns the whole thing already, so there's no laziness to preserve
  here; `show_image()` is just `show_stack()` with `array[np.newaxis]`
  (one frame).
- `show_stack_lazy(frame_getter, n_frames)`: `frame_getter(i)` is
  called for exactly the one frame currently on screen, initially and
  on every slider move -- never the whole stack. `MainWindow._open_project()`
  passes `DataInterface.get_projections` (bound, so `frame_getter(i)`
  is `get_projections(index=i)`) for this. `get_projections()` with no
  index -- what `_open_project()` used to call -- reads the entire
  stack regardless of `storage_mode`; `get_projections(i)` reads just
  that one h5py slice or one tiff file, for either format, is what
  makes disk mode's laziness actually hold on opening a project. A
  sinogram (one row across every angle) is a different access pattern
  with no equivalent shortcut for tiff -- building one still means
  opening every file, which is why `preprocessing.normalize_slice()`/
  `normalize_rows()` (used by calibration/reconstruction, not by the
  viewer) do exactly that when they need one.

Scrubbing the slider does not reset zoom/pan/contrast; loading a new
source (`_set_source()`) does (`reset_view=True` only on frame 0 of a
fresh source, passed through as `setImage()`'s `autoLevels`/`autoRange`).

No ROI tools (crop box, line profile, rotation-angle line) wired up
yet -- pyqtgraph has them, nothing in `gui/` uses them.

A `QLabel` under the slider reads out `x`/`y`/`value` under the
cursor, via `ImageView.scene.sigMouseMoved` ->
`ImageView.view.mapSceneToView()` -> a floor to pixel indices. Reads
from `ImageViewer._current_frame` (the frame `_show_frame()` most
recently displayed, cached specifically so hovering never triggers its
own disk read in lazy mode), not from `imageItem.image` -- pyqtgraph
stores that transposed (`(x, y)`) internally regardless of the `axes=`
used to display it, so indexing it directly would silently
re-introduce the row/column mixup `show_image()`'s explicit `axes=`
(still passed on every `setImage()` call, in `_show_frame()`) exists to
avoid.

## What isn't built yet

- Project panel status color. The checkbox now has one consumer
  (Reconstruct selected); copy-settings-between-projects would be a
  second.
- Streaming a reconstruction to disk slice by slice, so a volume
  larger than available RAM can still be reconstructed. Currently
  refused outright by the RAM check above rather than attempted.
- Dock layout persistence.
- Crop/clip/rotation controls, and any viewer-driven ROI control (crop
  box, line profile) -- the pixel-value readout is built (see Viewer,
  above), the rest of the ROI tools aren't.
- Refinements tab (ring/zinger removal).
- Import-from-config (loading a batch `.ini` directly into the project
  panel, rather than one project's `.ini` via Basic settings).
- `NoVendorMatchError`/`AmbiguousVendorMatchError` has no dedicated UI;
  it currently just lands in `session.error` the same as a missing
  completion field.
