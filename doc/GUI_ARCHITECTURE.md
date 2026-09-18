# GUI architecture

The desktop GUI for `tomo_project`, in `gui/`. Built on PySide6 and
pyqtgraph. Desktop only, no web interface.

`gui/` depends on `orchestration/` (and, through it, every other
package). Nothing outside `gui/` depends on it.

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

`MainWindow` is a `QMainWindow`: two `QDockWidget`s (Projects, left;
Adjustment, right), movable/floatable but not closable; `ImageViewer`
is the central widget. Menu bar (`File`, `Tools`) and status bar.

Dock layout is not persisted between launches; it resets to the
default each time.

## ProjectSession

One project as the GUI tracks it (`gui/session.py`):

- `config`: a `ProjectConfig`.
- `run`: the `PipelineRun` driving it (`None` until import starts).
  `session.config` and `session.run.config` are the same object.
- `checked`: membership in the next batch action.
- `preview_row`: which row is shown/reconstructed in the viewer.
  `None` until resolved to the middle row.
- `error`: a `PipelineStageError` from a failed `complete()` or
  `import_data()`, or `None`.
- `export_rows`: `(from, to)`, both inclusive, or `None` for the full
  range. GUI-only; translated into `config.rows` at export time.
- `export_crop_yx`: `(y0, y1, x0, x1)`, half-open, or `None`.
  GUI-only; translated into `config.crop` at export time.
- `exported`: set once a batch export (`run_batch()`) succeeds for
  this session.

`unique_project_id(stem, existing_ids)` disambiguates a new session's
`project_id` against every id already in the project panel.

`center`/`tilt_deg` on a fresh session's config are never left `None`:
`tilt_deg` is set to `0.0` at creation, `center` resolves to the
image's midpoint (`width / 2.0`) the first time
`AdjustmentPanel._refresh_from_session()` knows this project's width.

## Threading

Every pipeline call (import, calibrate, reconstruct, at any scope)
runs on a `QThreadPool` worker via `workers.run_in_background()`, never
on the GUI thread. `finished`/`failed` are Qt signals, delivered back
on the GUI thread. The caller keeps a reference to the returned
`Worker` until it fires (`AdjustmentPanel._workers`,
`MainWindow._workers`).

`MainWindow` shows a busy indicator (an indeterminate `QProgressBar` in
the status bar) for as long as any background operation is in flight,
via `_push_busy()`/`_pop_busy()` (a counter, not a boolean).

## Import

`File > Import` and `File > Import to RAM` both open `ImportDialog`,
differing only in the `storage_mode` passed to it. The dialog stages
paths (`Open File`: multi-select hdf5; `Open Folder(s)`: multi-select
tiff project folders) into a list; `Remove` drops a staged entry;
`Import` commits the whole list. Selecting a folder is always exactly
one tiff project.

On commit, `MainWindow._import()` builds one `ProjectConfig` and
`ProjectSession` per staged item, adds it to the project panel, and
starts `import_data()` + `complete()` for it in the background
(`_import_session()`). A `PipelineStageError` from `complete()` (e.g.
missing `geometry_type`) is stored on `session.error`, not raised; the
project still shows up and stays selectable, and the missing field can
be filled in via Basic settings before the next reconstruct attempt
retries `complete()`.

### Plugin picker

`import_data()` failing wraps every import-time exception
(`NoVendorMatchError`, `AmbiguousVendorMatchError`,
`ProjectImportError`, `ValueError`) into one `PipelineStageError`,
keeping the original as `.__cause__`.

`PluginPickerDialog` (`gui/plugin_picker_dialog.py`) is offered when
the cause is `NoVendorMatchError`/`AmbiguousVendorMatchError`. It lists
`get_plugins(session.config.format)` by name; picking one sets
`session.config.plugin_name` and retries `_import_session()`. Cancel
calls `ProjectPanel.remove_session()`.

If the retry also fails, the picker is offered again regardless of the
new exception type. Picking a different plugin can only ever resolve
an `AmbiguousVendorMatchError`, not a true `NoVendorMatchError`: the
low-specificity fallback plugins (`generic_hdf5`/`generic_tiff`) match
any hdf5 file with a 3D dataset or any tiff folder with at least one
file, so a `NoVendorMatchError` means no plugin recognizes the file's
data at all.

## Project panel

`ProjectPanel` is a `QTreeWidget`. One top-level item per session
(`project_id`, checkable), with child rows underneath it for Status,
Center of rotation, Tilt, Format, Storage (`INFO_FIELDS`).
`refresh(session)` updates an existing item's text in place, keyed by
`id(session)` in `ProjectPanel._items`. `MainWindow` connects
`AdjustmentPanel.session_changed` to `refresh()`.

Three independent states per row:

- **checked**: the top-level item's checkbox. `checked_sessions()`
  returns every checked session.
- **selected**: `QTreeWidget`'s native row selection.
- **opened**: double-clicking the top-level item or any child emits
  `project_opened`; `MainWindow._open_project()` loads that project
  into the viewer and points the adjustment panel at it.

### Status color

`_status_of(session)` / `_STATUS_COLORS` color the Status child and
the project name:

| State | Color | Condition |
|---|---|---|
| importing | blue | `session.run is None`, `session.error is None` |
| error | red | `session.run is None`, `session.error` set |
| needs_input | orange | `project.needs_user_input()` |
| ready | gray | resolved, not yet reconstructed |
| reconstructed | green | `run.volume is not None`, or `session.exported` |

`session.exported` is set by `MainWindow._reconstruct_selected()`'s
`done()` handler for each session `run_batch()` reports successful --
`run_batch()` builds its own internal `PipelineRun` per config and
never hands back the volume, so `session.run.volume` stays `None` even
after a successful batch export.

## Adjustment panel

`AdjustmentPanel` holds a `QTabWidget` (Basic settings, Automatic
adjustments, Refinements, Save settings) plus two buttons below it,
outside the tabs: Reconstruct (slice) and Reconstruct all slices. Both
always act on `self.session`, the currently open project, regardless
of which tab is active.

`set_session(session)` is the entry point that changes which project
the panel edits; it disables the whole panel when `session` is `None`.

### Basic settings

Fields: geometry type, angle span, pixel size, slice (z --
`session.preview_row`, not a `ProjectConfig` field), reconstruction
algorithm, number of iterations, device, center of rotation, tilt, and
a "Load config file..." button (`orchestration.load_config()` on a
single-project `.ini`, replacing `session.config` while keeping
`project_id`, and resetting `session.run` to `None`).

`angle_span`/`geometry_type` write into
`session.config.completion_answers` when changed -- these are
completion's "ask" fields, resolved here rather than in a separate
dialog. `angle_span_spin` is disabled, with a tooltip, whenever its
value is read from the file's own recorded angles rather than typed by
the user.

`pixel_size_spin` is disabled only when `DataInterface.get_metadata_source("voxel_size") == "file"`;
otherwise it's editable, writing into
`completion_answers["voxel_size"]`.

The reconstruction-algorithm combo lists every registered backend
except `algotom_gridrec` (needs `tomopy`, conda-only, no pip wheel --
usable via the CLI/scripting API instead). An "Iterations" spin box
(`num_iter_spin`, writing into
`session.config.reconstruction_options["num_iter"]`) is shown only for
the four iterative backends (SIRT/SART/ART/CGLS --
`ReconstructionBackend.is_iterative`); it's seeded with `50` for
SIRT/SART/CGLS, or `n_angles * width` for ART, the first time an
iterative backend is selected and no value is already set. Switching
backends again never overwrites an already-set value. A device combo
(`_DEVICE_CHOICES`: Auto/CPU/GPU) writes into `session.config.device`;
`reconstruction.resolve_device()` reports which device a reconstruction
actually used in the status message afterward.

The reconstruct status message states the unit (`1/cm`) and, when
`voxel_size`'s source isn't `"file"`, appends a caveat that the scale
may not be accurate.

### Automatic adjustments

**Find center of rotation**: `build_reference()` +
`normalize_slice()` on the stack's middle row, then `find_center()`.
Writes the result into `session.config.center` and the Basic settings
field.

**Find tilt**: sets `session.config.rows = None`, runs
`PipelineRun.preprocess()` on the full stack, then `find_tilt()` on
`run.normalized`. Writes `tilt_deg` the same way.

Both flag the result in the status message (not block it) if it falls
outside `MAX_PLAUSIBLE_COR_OFFSET_FRACTION` (2% of width from the
image's midpoint) or `MAX_PLAUSIBLE_TILT_DEG` (1 degree). Both also
trigger `self._reconstruct(all_slices=False)` once the new value is
written, so the viewer's preview updates immediately.

### Refinements

Placeholder tab. Ring/zinger removal
(`ProjectConfig.zinger_removal`/`ring_correction`) has no control here
yet.

### Save settings

Everything here affects `MainWindow._reconstruct_selected()`
("Reconstruct selected", the one action that exports) --
Adjustment's own two reconstruct buttons are preview-only. `dtype`,
`export_format`, and `stream` live on `session.config` directly; the
slice range and crop rect live on `ProjectSession`
(`export_rows`/`export_crop_yx`), translated into `config.rows`/
`config.crop` only when `_reconstruct_selected()` runs.

**Stream to disk (low RAM)**: `stream_checkbox`, writing into
`session.config.stream` -- see "Streaming" below.

**Clip min/max**: enabled only when `dtype == "uint16"`.
"Use preview slice's range" copies the last single-slice preview's own
min/max into the two fields. Leaving both fields at their default (0)
lets `to_uint16()` use the exported volume's own min/max instead,
computed at export time.

**From slice / to slice**: `session.export_rows`, both inclusive.
Resolves to the full range the first time a session's row count is
known. Converted to Python's half-open `slice(start, end + 1)` (or
`None` for the unrestricted case) only inside
`_reconstruct_selected()`.

**Crop**: a `pg.RectROI` (`ImageViewer._crop_roi`), hidden by default.
The checkbox shows it, seeding an 80%-of-frame default the first time,
or restoring the session's own stored rect otherwise. Dragging it
writes `(y0, y1, x0, x1)`, half-open, into `session.export_crop_yx`.
Four spin boxes ("Starting point"/"End point", x/y) mirror the box
both ways: dragging updates them, and typing into them moves the box
(`crop_rect_edited` -> `ImageViewer.set_crop_rect()`).
`get_crop_rect()` does not clamp against the viewer's currently
displayed frame -- an out-of-range crop is harmless; numpy slicing
clips it silently wherever `config.crop` is applied in
`postprocess()`.

Rotation (a post-reconstruction image rotation, distinct from
rotation-axis tilt) is not implemented.

### Reconstruct scope

| Action | Panel | Scope | Rows |
|---|---|---|---|
| Reconstruct (slice) | Adjustment | one project, current row | `slice(preview_row, preview_row + 1)` |
| Reconstruct all slices | Adjustment | one project, every row | `None` |
| Reconstruct selected | Project | every checked project | per project, from Save settings' `export_rows` |

Both adjustment-panel actions call `_run_pipeline_to_reconstruction()`:
`import_data()` if not already done, `complete()`, `preprocess()`,
`calibrate()`, `reconstruct()`. Neither exports; both only update
`self.viewer`.

`ProjectPanel.reconstruct_selected_btn` ("Reconstruct selected") is the
only action that exports. `MainWindow._reconstruct_selected()` passes
`checked_sessions()`'s configs to `orchestration.run_batch()` (the same
function `tomo batch` uses) in one background worker. Each config's
`export_path` is set at import time to
`<source file's directory>/<project_id>_volume.h5`.

Before dispatching, `_reconstruct_selected()` translates each checked
session's `export_rows`/`export_crop_yx` into that session's
`config.rows`/`config.crop`. A project still importing is skipped
before the batch starts and reported in the status bar. So is a
project whose reconstructed volume is estimated not to fit in
available RAM (see "RAM check" below), unless "Stream to disk" is
checked for it.

## RAM check

`AdjustmentPanel`'s own two reconstruct buttons and
`_reconstruct_selected()`'s non-streaming path go through
`reconstruct()`/`postprocess()`/`export()`, which take and return a
full in-memory array. `orchestration/ram_check.py`'s
`estimate_volume_bytes(n_rows, width)` assumes a square `(width,
width)` output slice at float32, times `SAFETY_FACTOR` (2.5), against
`psutil.virtual_memory().available`. Under that, the reconstruction is
refused before it starts. Skipped entirely for a session with "Stream
to disk" checked.

## Streaming

Save settings' "Stream to disk (low RAM)" checkbox writes into
`session.config.stream`, switching `_reconstruct_selected()`'s
`run_batch()` call to `PipelineRun._run_streaming()` for that project:
one chunk of rows at a time, sized by
`orchestration/ram_check.py`'s `estimate_chunk_rows()` or
`stream_chunk_rows` if set explicitly.

Requires `config.center` already resolved (always true here) and, for
a uint16 export, `dtype_clip_min`/`dtype_clip_max` set explicitly.
`_run_streaming()` raises `PipelineStageError` if either is missing.
See doc/RECONSTRUCTION.md's "Streaming" section for the full contract.

## Viewer

`ImageViewer` wraps a `pyqtgraph.ImageView`, replacing its built-in
frame slider with its own (`_frame_slider`/`_frame_label`), through one
shared path (`_set_source()`/`_show_frame()`/`_frame_at()`):

- `show_stack(array)` / `show_image(array)`: an eager, fully-in-memory
  numpy array (a reconstructed volume/slice). `show_image()` is
  `show_stack()` with `array[np.newaxis]`.
- `show_stack_lazy(frame_getter, n_frames)`: `frame_getter(i)` is
  called for exactly the one frame on screen, initially and on every
  slider move. `MainWindow._open_project()` passes
  `DataInterface.get_projections` (bound) for this --
  `get_projections(index=i)` reads just one h5py slice or one tiff
  file, for either storage mode.

Scrubbing the slider does not reset zoom/pan/contrast; loading a new
source does.

A `QLabel` under the slider reads out `x`/`y`/`value` under the
cursor, from `ImageViewer._current_frame` (the most recently displayed
frame, cached so hovering doesn't trigger a disk read in lazy mode).

## What isn't built yet

- Copy-settings-between-projects.
- Dock layout persistence.
- Rotation control (post-reconstruction image rotation).
- Refinements tab (ring/zinger removal).
- Import-from-config (loading a batch `.ini` directly into the project
  panel).
- A UI for `source_to_detector_distance`/`source_to_object_distance`
  (required, `ask`-only, only when `geometry_type == "cone"`).
- No cross-session guard on background operations: if the open project
  changes while a reconstruct/find-COR/find-tilt worker for the
  previous one is still in flight, its result still updates the
  viewer/Basic settings for whichever project is open when it lands.
- `ProjectObject.close()` (releases a disk-mode project's open
  `h5py.File`) is never called.
