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
  metadata_toolbox.py    MetadataToolbox (dock widget); metadata_info.py is its Qt-free model
  session.py             ProjectSession, unique_project_id()
  workers.py             Worker, run_in_background()
```

## Window layout

`MainWindow` is a `QMainWindow`: two `QDockWidget`s (Projects, left;
Adjustment, right), movable/floatable but not closable; `ImageViewer`
is the central widget. Menu bar (`File`, `Tools`) and status bar.
Tools > Convert TIFF project to HDF5... opens `ConvertDialog`
(`gui/convert_dialog.py`); see [`CONVERTER.md`](./CONVERTER.md).
Tools > Metadata toolbox shows or hides a third dock (see below), which, unlike the
other two, can be closed.

## Metadata toolbox

A dock (hidden at start, left side) that lists every metadata field of one project
(`gui/metadata_toolbox.py`; the rows come from `gui/metadata_info.py`, which has no Qt).
It follows the project selected in the project panel and the one opened in the viewer,
and updates when the project changes (import finished, completion answers entered).

Rows are grouped by tier (see [`PROJECT_METADATA.md`](./PROJECT_METADATA.md)): must be in
the file, needed (can be filled in), optional, calculated by the software, and other (the
plugin's own fields and `extra.*`). Each row shows the field, its value (an array as its shape and
range), a status and the unit:

| Status | Meaning |
|---|---|
| read from file | the plugin read it |
| set by you | a completion answer you gave |
| default | a fixed fallback was applied (check it) |
| derived | computed from other fields |
| calculated | the software calculated it (centre, tilt) |
| set in settings | centre or tilt typed in the settings, not yet calculated |
| MISSING (red) | not in the file and not filled in; the tooltip says how it can be resolved |
| not yet calculated | a calculated field, before calibration has run |
| not needed | only for another geometry (the cone-beam distances in a parallel project) |

The line above the list counts them. When completion could not fill in a required field,
the same line carries its message, so the toolbox also shows why a project cannot continue.
"Only what is missing or not from the file" hides the fields that were simply read.

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

Double-clicking the empty area below the project rows opens the import dialog
(from disk); right-clicking it offers *Import project...* and *Import project
to RAM...*.

`ProjectPanel` is a `QTreeWidget`. One top-level item per session
(`project_id`, checkable), with child rows underneath it for Status,
Center of rotation, Tilt, Format, Storage (`INFO_FIELDS`).
`refresh(session)` updates an existing item's text in place, keyed by
`id(session)` in `ProjectPanel._items`. `MainWindow` connects
`AdjustmentPanel.session_changed` to `refresh()`.

Three independent states per row:

- **checked**: the top-level item's checkbox. `checked_sessions()`
  returns every checked session.
- **selected**: `QTreeWidget`'s native row selection, extended
  (multi-select); only top-level (project) rows are selectable, not
  their `INFO_FIELDS` children.
- **opened**: double-clicking the top-level item or any child emits
  `project_opened`; `MainWindow._open_project()` loads that project
  into the viewer and points the adjustment panel at it.

### Right-click menu

Right-clicking a project row selects it first if it wasn't already
part of the current selection, then shows a menu built from
`self.tree.selectedItems()`:

- **Copy settings** / **Copy selected settings...**: only shown for a
  single-project selection (copying more than one project's settings
  at once has no single meaning). "Copy settings" copies every field
  in `gui/settings_clipboard.py`'s `COPYABLE_SETTINGS` immediately;
  "Copy selected settings..." opens `CopySettingsDialog`
  (`gui/copy_settings_dialog.py`) first -- one checkable `QGroupBox`
  per category, individual checkboxes underneath, global Select
  all/Deselect all -- and copies only what's checked on accept. Both
  write into `ProjectPanel._clipboard`, a single in-memory `dict`
  (`{field_name: value}`), not saved anywhere. The "Scan" category
  (geometry type, angle span, pixel size) copies only values that were
  typed in. Values a project loaded from its own file are neither
  copied nor overwritten on paste.
- **Paste settings**: applies `_clipboard` onto every selected
  session via `settings_clipboard.paste_settings()`, refreshing each
  row immediately. Disabled while `_clipboard is None`. Works for a
  multi-selection.
- **Export config(s)**: `orchestration.save_config()` for every
  selected session, each to its own `{project_id}_config.ini` next to
  its data file, no prompt. Works for a multi-selection (labeled
  "Export configs" then).
- **Export config to...**: only shown for a single-project selection
  (one chosen path can't serve more than one project) -- a Save File
  dialog, defaulting to the same `{project_id}_config.ini` path.
- **Delete project(s)**: after a confirm dialog, calls
  `remove_session()` for every selected session -- drops it from
  `self.sessions`/the tree, freeing the Python references (there is no
  `ProjectObject.close()` yet); the underlying data file is untouched.

`sessions_removed`/`settings_pasted` signals carry the affected
sessions to `MainWindow`, which clears the adjustment panel/viewer if
the currently-open session was deleted, or re-applies `set_session()`
if it was pasted onto (so the Basic/Refinements/Save settings fields
stay in sync with what just changed).

`COPYABLE_SETTINGS` only lists fields that don't come from the
project's own file (center, tilt, rotate, reconstruction backend/
device/options, ring/zinger settings, output dtype/clip, crop
rectangle, export format, stream) -- `project_id`/`path`/`format`/
`completion_answers` (geometry_type, angle_span, voxel_size, ...) are
never copyable. `"crop_yx"` is the one key mapped onto
`ProjectSession.export_crop_yx` instead of a `ProjectConfig` field.

### Status color

`_status_of(session)` / `_STATUS_COLORS` color the Status child and
the project name:

| State | Color | Condition |
|---|---|---|
| importing | blue | `session.run is None`, `session.error is None` |
| error | red | `session.run is None`, `session.error` set |
| needs_input | orange | `project.needs_user_input()` |
| ready | gray | resolved ("ready for reconstruction"); a single-slice preview does not change this |
| reconstructed | green | `session.exported`, `volume_path` or `ram_volume` set (a full reconstruction) |

`session.exported` is set by `MainWindow._reconstruct_selected()`'s
`done()` handler for each session `run_batch()` reports successful --
`run_batch()` builds its own internal `PipelineRun` per config and
never hands back the volume, so `session.run.volume` stays `None` even
after a successful batch export.

## Status box

`StatusBox` (`gui/status_box.py`), always visible at the bottom of the
project panel: project, stage, device (CPU/GPU), a progress bar and
"N / total slices reconstructed, M left". Fed by `orchestration.ProgressEvent`s:
`PipelineRun(config, progress=callback)` and `run_batch(configs, progress=...)`
report each stage (importing, completing, normalizing, calibrating,
reconstructing, postprocessing, writing); a streamed run also reports
slices done per chunk. Events come from worker threads, so the GUI passes a
`ProgressRelay` (a Qt signal) as the callback. Stages without a slice count
(eager runs) show an indeterminate bar. The status bar still shows
one-line messages.

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
algorithm, number of iterations, device, center of rotation, tilt,
rotate, and three config-file buttons: "Load config file..."
(`orchestration.load_config()` on a single-project `.ini`, replacing
`session.config` while keeping `project_id`, and resetting
`session.run` to `None`), "Save config" (`orchestration.save_config()`,
writing next to the project's own data file as
`{project_id}_config.ini`, no prompt -- repeatable, like Ctrl+S), and
"Save config as..." (a Save File dialog for an explicit path).

**Allow editing values loaded from the project**: checkbox at the top of
Basic settings. When checked, angle span and pixel size values read from
the project's file can be edited. Edits are stored in
`session.config.completion_answers`.

`angle_span`/`geometry_type` write into
`session.config.completion_answers` when changed -- these are
completion's "ask" fields, resolved here rather than in a separate
dialog. `angle_span_spin` is disabled, with a tooltip, whenever its
value is read from the file's own recorded angles rather than typed by
the user.

`pixel_size_spin` is disabled only when `DataInterface.get_metadata_source("voxel_size") == "file"`;
otherwise it's editable, writing into
`completion_answers["voxel_size"]`.

Choosing **cone** as the geometry type shows the two distances (source-detector, source-object, mm; stored as
completion answers) and restricts the algorithm list to cone-beam backends (see [`CONE_BEAM.md`](./CONE_BEAM.md)); switching
back to parallel restores the usual list. "Reconstruct (slice)" is not available for cone-beam data.

The reconstruction-algorithm combo is built from the backend registry
(`reconstruction.list_backends(geometry=...)`): every registered backend that supports the project's geometry and whose
required packages are installed, in a fixed order (`algotom_fbp`,
`gridrec`, `algotom_bpf`, ...) with any others alphabetically after.
`algotom_gridrec` (needs conda-only tomopy) and the ASTRA backends
(need `astra-toolbox`) therefore only appear when those are installed,
and a backend someone adds shows up with no change here. An "Iterations" spin box
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

`rotate_spin` writes into `session.config.rotate_deg` (degrees, `0.0` =
no rotation). Every reconstruct call goes through
`PipelineRun.reconstruct()`, so the preview buttons here and
"Reconstruct selected" both pick it up automatically -- see
doc/RECONSTRUCTION.md's "Rotation" section for the sinogram-reorder/
image-domain-fallback split.

### Automatic adjustments

**Find center of rotation** searches the middle row. Fully automatic by
default; type a **Guess** (or "Use current center") and a **Search +/-**
tolerance to search only that window (a guess alone means +/- a tenth of the
width). The best few candidates are listed ("match 1.00" is the winner; a
second one near 1.0 means the answer is ambiguous): click one to use and
preview it. If the best match sits at the edge of the window, the message and
the note under the list say the true center may lie outside it. Guess,
tolerance and candidates are kept per project.
(`calibration.search_center()`, see CALIBRATION.md.)

**Auto-update preview** (Basic settings, under Center of rotation): every
change of the center (mouse wheel, typing, a candidate) re-previews the slice
after a 120 ms pause. `PreviewEngine` (`gui/preview_engine.py`) remembers the
normalized sinogram of the slice and every preview image (LRU, 512 MB), so a
center already tried shows instantly, and with auto-update on the centers 3
wheel-steps either side (and the other candidates after a search) are computed
in the background. A change while a preview is computing is answered with the
latest center when it finishes. One wheel notch is 0.5 px.

**Find tilt** is unchanged.

### Refinements

`preprocessing.remove_ring_artifacts()`/`remove_zinger()`
(`preprocessing/corrections.py`), applied in `PipelineRun.preprocess()`
and the streaming per-chunk loop -- like Basic settings, changes here
affect the preview reconstruct buttons immediately, not just
"Reconstruct selected".

**Projections already flat/dark corrected**: checkbox writing
`session.config.projections_normalized`. When checked, the flat/dark
division is skipped in Find COR, Find tilt, previews and
reconstruction.

**Ring correction**: checkbox writing `session.config.ring_correction`,
plus a method dropdown (`session.config.ring_method`) enabled only
while checked -- all 12 algotom stripe-removal variants
`preprocessing/corrections.py`'s `RingMethod` wraps (`sorting`,
`filtering`, `fitting`, `normalization`, `all`, `large_stripe`,
`dead_stripe`, `interpolation`, `2d_filtering_sorting`, `fft`,
`wavelet_fft`, `regularization`), each with a one-line tooltip. `ring_options`
(backend-specific kwargs) has no GUI control, same as the INI grammar
-- code-only.

**Zinger removal**: checkbox writing `session.config.zinger_removal`,
plus threshold/size spin boxes (`zinger_threshold`/`zinger_size`),
enabled only while checked. Left unedited, both stay `None` on
`config` (`PipelineRun.preprocess()`'s own defaults, `0.08`/`2`, apply)
-- the spin boxes only show those defaults, the same "display a
fallback without writing it back" convention `tilt_spin` uses for
`tilt_deg=None`.

**Phase retrieval** (a group at the bottom of the tab; see
[`PHASE_RETRIEVAL.md`](./PHASE_RETRIEVAL.md)): a checkbox, the method, and four
values -- delta/beta, propagation distance, energy, pixel size. Each shows the
value the project already provides (the file's energy and distance, or one
typed in as a completion answer; the pixel size from the file or Basic settings,
never a silent default), with a small note on where it came from ("from the
file", "typed in", "set here"). Typing a value stores it in
`ProjectConfig.phase_*` and overrides the project's; setting it back to 0
("not set") returns to the project's value. delta/beta has no default. Under the
controls, one line says what is still needed (in red) or, once complete, the
smoothing length in um and pixels and the halo in rows
(`gui/phase_info.py`, Qt-free). Changing any value re-previews the slice when
"Auto-update preview" is on and the settings are complete; the settings take part
in the preview cache key, so each delta/beta already tried is remembered. All
phase fields can be copied between projects (own "Phase retrieval" category in
Copy settings). If the settings are incomplete when a reconstruction starts,
the run stops with an error naming what is missing.

**Beam hardening** (the group below phase retrieval; see
[`BEAM_HARDENING.md`](./BEAM_HARDENING.md)): a checkbox and the coefficients
`c1`, `c2`, `c3` of `p_corrected = c1 p + c2 p^2 + c3 p^3`. Ticking starts from
(1, 0, 0), which changes nothing. The line under the fields shows what the values
do to attenuations of 1, 2 and 3, or, in red, why they are unusable (the
polynomial must be increasing at least up to attenuation 3); an unusable
set never triggers a preview. With "Auto-update preview" on, each edit
re-previews the slice. Both settings can be copied between projects (the "Beam
hardening" category in Copy settings).

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

### Reconstruct scope

| Action | Panel | Scope | Rows |
|---|---|---|---|
| Reconstruct (slice) | Adjustment | one project, current row | `slice(preview_row, preview_row + 1)` |
| Reconstruct all slices | Adjustment | one project, every row | `None` |
| Reconstruct selected | Project | every checked project | per project, from Save settings' `export_rows` |

**Reconstruct all slices** first asks where the volume should go
(`ReconstructDestinationDialog`, numbers from `gui/reconstruct_plan.py`):
it shows available RAM against the estimated need (normalized
sinograms plus volume, with `SAFETY_FACTOR`), and offers

- *Write to an HDF5 file (streamed)*: runs `PipelineRun.stream()` on the
  already-imported project, one chunk of slices at a time, so the volume
  never sits in RAM. Parallel-beam only (cone-beam slices need
  neighbouring detector rows, so row chunking doesn't apply); the option is
  disabled otherwise. A uint16 setting without Clip min/max is written
  as float32. The chosen folder is remembered (`RememberedPaths`, in
  memory only) as the default for later reconstructions.
- *Keep in memory only*: the old path; disabled when it doesn't fit.

Which slices, and in what type: **Reconstruct all slices** uses Save
settings' From/To slice range (inclusive) and Output dtype, which default
to every slice and float32. The dialog states both. The HDF5 file is
written in that dtype; uint16 needs Clip min/max for streaming, and
without them the file is written as float32 (the in-memory path scales to
the volume's own range). The crop box is not applied here.

If neither fits (e.g. cone-beam, low RAM) a message box says so. A
`MemoryError` during a run is reported by name instead of an empty
message. Single-slice previews are unchanged.

When a full reconstruction exists (streamed file, in-memory volume, or a
`Reconstruct selected` HDF5 export) the project's **Reconstructed** row
shows it; double-clicking that row shows the volume in the viewer
(slice by slice from the file) instead of the projections.

Both preview actions call `_run_pipeline_to_reconstruction()`:
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

**z / y / x buttons** above the image choose the slicing direction
(`gui/slicing.py`); they look like push buttons but act as radio buttons
(the chosen one stays pushed). z (default) is the stack's first axis, y
and x are the other two. For raw projections `(angle, row, column)` that
makes y the **sinogram** of one detector row and x an angle-by-row image;
for a volume they are the xz and yz sections. A new source always starts
in z, and a y/x switch starts at the middle section. They need the
stack's full shape (`show_stack_lazy(..., shape=...)`), so a single
preview slice only has z. The crop box is an xy rectangle: it is drawn
only in the z view, and turning crop on switches back to z. TIFF
projections read every file for a y/x section, so scrubbing those is slow.

Scrubbing the slider does not reset zoom/pan/contrast; loading a new
source does.

A `QLabel` under the slider reads out `x`/`y`/`value` under the
cursor, from `ImageViewer._current_frame` (the most recently displayed
frame, cached so hovering doesn't trigger a disk read in lazy mode).

## What isn't built yet

- Dock layout persistence.
- Import-from-config (loading a batch `.ini` directly into the project
  panel).
- No cross-session guard on background operations: if the open project
  changes while a reconstruct/find-COR/find-tilt worker for the
  previous one is still in flight, its result still updates the
  viewer/Basic settings for whichever project is open when it lands.
- `ProjectObject.close()` (releases a disk-mode project's open
  `h5py.File`) is never called.
- Reconstructed row: no "choose one path for all selected projects" yet.
