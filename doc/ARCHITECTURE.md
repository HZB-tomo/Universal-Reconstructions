# Architecture

How the codebase is organized and how data flows through it.

## The big picture

Seven packages:

```
core/            the shared data model
data_import/     turns a file on disk into a ProjectObject
completion/      fills in metadata a vendor's file didn't provide
preprocessing/   turns raw projections into reconstruction-ready sinograms
calibration/     finds center-of-rotation and rotation-axis tilt
reconstruction/  turns sinograms into reconstructed images
orchestration/   drives one or many projects through every stage above,
                 configured by a ProjectConfig; see ORCHESTRATION.md
```

`core/` defines a loaded dataset (`ProjectObject`), how data is read
out of one (`DataInterface`), how data is stored (`DataHandle`), how
metadata is tracked (`MetadataBundle`), and the catalog of metadata
fields the software depends on (`known_fields.py`). `core/` has no
knowledge of files, hdf5, tiff, or vendors.

`data_import/` produces a `ProjectObject`: opens hdf5/tiff files,
identifies the vendor that wrote a file, and reads that vendor's
metadata into the shape `core/` expects.

`completion/`, `preprocessing/`, `calibration/`, and `reconstruction/`
are the pipeline stages after import; see
[`USING_THE_PIPELINE.md`](./USING_THE_PIPELINE.md). `completion/` and
`preprocessing/` depend only on `core/`: neither imports from
`data_import/`, both work with a `ProjectObject` via `DataInterface`.

`calibration/` and `reconstruction/` do not depend on `core/` and do
not take a `ProjectObject`; see their sections below.

`orchestration/` is the only package allowed to depend on all the
others -- it chains import through export into one run. None of the
packages above depend on it back; each is usable stage-by-stage
without it, as `USING_THE_PIPELINE.md` describes.

## `core/` in detail

### `MetadataBundle` (`core/metadata.py`)

Metadata about a dataset (voxel size, angles, center of rotation,
operator name, etc.) is stored as a `FieldSpec`: a value plus one of
four tiers:

| Tier | Meaning | Example |
|---|---|---|
| `mandatory_hard` | Missing -> import fails | projections |
| `mandatory_soft` | Missing -> import succeeds, project flagged as needing user input | voxel size, geometry type |
| `optional` | No consequence if missing | operator name |
| `computed` | Filled in later by an algorithm | center of rotation |

`MetadataBundle` does not know about hdf5 or tiff; vendor plugins
assign each field's tier.

`missing_hard()` reports whether a dataset is usable. `missing_soft()`
reports whether the user needs to be prompted.

`computed` fields are overwritten, not versioned: a new calculated
value replaces the previous one.

### `DataHandle` (`core/handles.py`)

Array data (projections, flats, darks) is wrapped in a `DataHandle`:

- `RamHandle`: a numpy array in memory.
- `Hdf5DiskHandle`: a reference to an open `h5py.Dataset`, read from
  disk on access.
- `TiffDiskHandle`: a sorted list of per-slice file paths; `.shape`
  reads only the first file's header, `.get()` reads only the files it
  needs.

All three expose the same interface (`.get()`, `.shape`,
`.storage_mode`, `.materialize()` -> a `RamHandle`; a no-op on
`RamHandle` itself). Both formats support `storage_mode="ram"` and
`storage_mode="disk"`.

### `ProjectObject` (`core/project.py`)

Represents one loaded dataset. Holds:

- `format`: `"hdf5"` or `"tiff"`
- `storage_mode`: `"ram"` or `"disk"`
- `metadata`: a `MetadataBundle`
- `data_handles`: a dict of `DataHandle`s, keyed `"projections"`,
  `"flats"`, `"darks"`
- `vendor_plugin_name`: which plugin produced this project
- `source_info`: free-form dict, e.g. the original file path

Exposes `is_ready_for_import()` and `needs_user_input()`.

`export_metadata()` flattens every resolved metadata field into a
plain dict (excluding fields ending in `_path`, which name a location
inside the source file and are meaningless once detached from it). For
feeding into `reconstruction.save_volume()`'s `settings`; see
`RECONSTRUCTION.md`. `reconstruction/` does not import `core/` for
this: the flattening happens on the `ProjectObject` side, so the
caller passes a plain dict either way.

Created only by `DataImporter`.

### `DataInterface` (`core/data_interface.py`)

Facade for reading data from a `ProjectObject`: `get_projections()`,
`get_flats()`, `get_darks()`, `get_metadata()`, `set_computed()`.
Returns a numpy array regardless of whether the project is disk-backed
or RAM-backed.

`materialize_to_ram()` converts every disk-backed handle in a project
to a RAM handle.

### `known_fields.py` (`core/known_fields.py`)

The catalog of every metadata field the software depends on: tier,
unit, and resolution strategy (`derive`, `default`, `ask`, or
`calculate`).

`reconcile(bundle)` fills in any known field a plugin did not mention,
using the registry's tier. Called from `data_import`'s
`import_project()` after a plugin's `parse_metadata()` returns.
`status(bundle)` reports the current state of every known field.

See [`PROJECT_METADATA.md`](./PROJECT_METADATA.md) for the field-by-field
reference.

## `data_import/` in detail

Given a file path, produces a `ProjectObject`. Three concerns:

```
"How do I recognize this vendor's file?"    -> signature.py (a plugin's `signature`)
"Where does this vendor put things?"        -> a plugin's get_path_map() / parse_metadata()
"How do I physically read hdf5/tiff?"       -> backend.py
```

### Signatures and detection (`data_import/signature.py`)

A `Signature` is a checklist of conditions (required hdf5 paths and
attribute values, or for tiff, filename patterns and sidecar file
requirements). Each condition holds or does not.

A signature's specificity is the number of conditions it checks.
Detection:

1. Every registered plugin's signature is checked against the file.
2. If none match: `NoVendorMatchError`, listing the failed conditions
   for every plugin.
3. Among matching plugins, the one with the highest specificity wins.
4. If two or more plugins tie for highest specificity:
   `AmbiguousVendorMatchError`.

The built-in fallback plugins use low specificity, so they only match
when nothing more specific does.

### The plugin contract (`data_import/plugin.py`)

A vendor plugin subclasses `ImportPlugin` and implements:

- `signature`: a class attribute, `Hdf5Signature` or `TiffSignature`.
- `get_path_map(handle)`: returns a `PathMap` pointing to
  projections/flats/darks/angles.
- `parse_metadata(handle, path_map)`: reads metadata into a
  `MetadataBundle`, sorted into the four tiers.

A plugin does not open a file itself (`backend.py`'s job) and does not
choose RAM versus disk (the caller's choice, via `DataImporter`).

Plugins register with `@register_plugin` when their module is
imported; `data_import/vendors/__init__.py` imports every vendor
module. The registry (`get_plugins()`, `clear_registry()`) is a plain
in-memory list per format.

This plugin pattern is scoped to `data_import/` only.

### Format backends (`data_import/backend.py`)

`Hdf5Backend` and `TiffBackend` read their format, with no knowledge of
vendor conventions. Each backend is reused by every plugin of that
format.

### `DataImporter` (`data_import/importer.py`)

`detect()` identifies which plugin should handle a file, without
loading data. `import_project()`:

1. Opens the file/folder via the right backend.
2. Detects (or uses a caller-supplied) plugin.
3. Calls the plugin's `get_path_map()` and `parse_metadata()`.
4. Raises `ProjectImportError` if a `mandatory_hard` field is missing.
5. Reads the arrays at the paths the plugin pointed to, building
   `DataHandle`s in the requested storage mode.
6. Assembles and returns the `ProjectObject`.

### `vendors/` — one file per vendor

Each file is a real vendor plugin, or (`generic.py`) the low-specificity
fallback pair. See `WRITING_A_PLUGIN.md`.

## `completion/` in detail

Drives every `mandatory_soft` field toward resolved, using the
strategy `core/known_fields.py` specifies. Two entry points:

- `run(project)`: tries `derive` then `default`, in order, for every
  currently-missing field, and repeats until a pass makes no further
  progress. Does not attempt `ask` or `calculate`.
- `provide(project, key, value)`: records an externally-collected
  answer (`source="user"`), then re-runs the automatic pass.

Does no asking itself. `core/known_fields.status(project.metadata)`
reports what remains.

Derive logic (`completion/derivers.py`) is separate from the pass/loop
logic (`completion/engine.py`). Built-in derivers: `angle_span` (from
`angles`), `angles` (from `angle_span`, projection count, and
`rotation_direction`), `detector_pixel_size` (from `voxel_size`). New
derivers register via `@register_deriver("field_name")`.

`completion/` does not touch `computed`-tier fields
(`center_of_rotation`, `tilt_angle`); those are `calibration/`'s job.

## `preprocessing/` in detail

Turns a completion-ready project's raw projections into
reconstruction-ready sinograms.

- `normalization.py`: flat/dark correction,
  `(raw - dark) / (flat - dark)`. Pure numpy, no optional dependency.
  `build_reference(project)` reduces the flats/darks stacks to one
  reference (a scalar offset for dark if the project has no dark
  field; `MissingFlatFieldError` if flats are absent). `normalize()`
  handles a single sinogram, a batch of rows, or the whole volume by
  broadcasting; `normalize_slice()`/`normalize_rows()` are
  `DataInterface`-based wrappers. Does not apply `-log()`.
- `corrections.py`: zinger removal and ring-artifact removal (four
  interchangeable algorithms plus a combined method), backed by the
  optional `algotom` package. `process_volume()` runs a single-sinogram
  function once per row in parallel, via algotom's
  `parallel_process_slices()`. Each function imports `algotom` lazily
  and raises `ImportError` if it is missing.

`preprocessing/` functions are pure: array in, array out. Nothing is
written back onto `ProjectObject`.

## `calibration/` in detail

Finds center-of-rotation and rotation-axis tilt from normalized
projections. Structured like `reconstruction/`: `backend.py`
(`CalibrationBackend`, one method, `find_center()`, plus a
`register_backend`/`get_backend` registry), `backends/algotom_cor.py`
(the built-in backend), `api.py` (`find_center()`, `find_tilt()`,
`centers_for_rows()`).

Does not take a `ProjectObject`. `find_tilt()` never reconstructs
anything; it is deterministic sinogram-domain math (independent
per-row `find_center()` calls plus a line fit). See
[`CALIBRATION.md`](./CALIBRATION.md).

## `reconstruction/` in detail

Turns normalized sinograms into reconstructed images. Does not take a
`ProjectObject`; `reconstruct()` takes plain arrays. See
[`RECONSTRUCTION.md`](./RECONSTRUCTION.md).

- `backend.py`: `Geometry` (parallel/cone, field names matching
  `core/known_fields.py`) and `ReconstructionBackend`, the contract:
  one method, `reconstruct(sinograms, angles, center, geometry, device, **options)`.
  Registered via `@register_backend`, looked up via `get_backend(name)`.
- `api.py`: the public `reconstruct()`. Input is
  `(n_angles, n_slices, width)`, matching
  `preprocessing.normalize_rows()`'s output; output is
  `(n_slices, height, width)`, matching this codebase's volume
  convention.
- `backends/algotom_fbp.py`: the built-in backend, parallel-beam only,
  wrapping `algotom.rec.reconstruction.fbp_reconstruction` (one
  sinogram at a time; this backend loops over the stack).
- `postprocess.py`: `to_uint16()`, clipping and rescaling a volume to
  16-bit before export.
- `export.py`: `save_volume()`, writing a reconstructed volume to disk
  as a TIFF stack or an HDF5 dataset, with an optional JSON
  settings/provenance record.

The contract specifies shapes in and out only; how a backend processes
a batch (loop, native batch call, CPU, GPU) is internal to the backend.

## `orchestration/` in detail

Drives one project through every stage above, or many independently.
See [`ORCHESTRATION.md`](./ORCHESTRATION.md).

- `config.py`: `ProjectConfig` (one project's settings, grouped by
  stage) and `PipelineStageError`, the one exception type every stage
  raises.
- `pipeline.py`: `PipelineRun`, whose seven stage methods
  (`import_data()`, `complete()`, `preprocess()`, `calibrate()`,
  `reconstruct()`, `postprocess()`, `export()`) are each independently
  callable and store their result on the instance; `run_all()` chains
  all seven and returns a `PipelineResult` instead of raising.
  `calibrate()` is the one stage that writes back onto the imported
  `ProjectObject`: `center_of_rotation`/`tilt_angle`, via
  `MetadataBundle.set_computed()`.
- `config_ini.py`: `load_config()`/`write_config_template()`/
  `resolve_config()`, reading/writing `ProjectConfig` as hand-editable
  INI text, single-project or batch grammar.
- `batch.py`: `validate_config()` (a cheap dry-run through calibration,
  advisory only) and `run_batch()` (runs every config independently;
  one project's failure, predicted by validation or not, never stops
  the rest).
- `cli.py`: the `tomo` command (`run`/`batch`/`validate`/`init-config`),
  registered via `pyproject.toml`'s `[project.scripts]`. A thin
  wrapper -- argument parsing and printing only, no logic of its own
  beyond converting an exception into a clean one-line error instead
  of a traceback.

Never blocks on input: with `interactive=False` (the default, and
always the case for `run_batch()`), a stage that can't resolve
something it needs raises `PipelineStageError` rather than prompting.

## Data flow, end to end

```
file path
  |
  v
backend.open()              (data_import/backend.py)
  |
  v
plugin.matches()  xN        every registered plugin checks its signature
  |
  v
highest specificity wins    (or NoVendorMatchError / AmbiguousVendorMatchError)
  |
  v
plugin.get_path_map()       where are projections/flats/darks/angles?
  |
  v
plugin.parse_metadata()     read metadata, sort into 4 tiers
  |
  v
known_fields.reconcile()    catch any known field the plugin never mentioned
  |
  v
backend.build_data_handle() x (projections, flats, darks)
  |
  v
ProjectObject assembled     (returned by DataImporter.import_project())
  |
  v
DataInterface(project)      everything downstream reads through here
  |
  v
completion.run() / .provide()   drive mandatory_soft fields to resolved
  |
  v
preprocessing.normalize() / remove_zinger() / remove_ring_artifacts()
  |
  v
calibration.find_center() / find_tilt() / centers_for_rows()   -> per-row centers
  |
  v
reconstruction.reconstruct()    sinogram(s) -> image(s), via the chosen backend
  |
  v
reconstruction.save_volume()    write the result to disk (TIFF stack or HDF5)
```

`orchestration.PipelineRun` runs this same sequence end to end for one
`ProjectConfig`; `orchestration.run_batch()` runs it independently for
many.
