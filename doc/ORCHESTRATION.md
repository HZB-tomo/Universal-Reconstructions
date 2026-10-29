# Orchestration

`orchestration/` drives one or many projects through the full pipeline
(import, completion, preprocessing, calibration, reconstruction,
postprocess, export), configured by a `ProjectConfig` built directly in
code, loaded from an INI file, or both.

`PipelineRun` never blocks on input. With `interactive=False` (the
default, and always the case for batch runs), a missing required value
raises `PipelineStageError`. With `interactive=True`, a single missing
completion field is prompted for on stdin. All other interactivity
(GUI dialogs, calibration review) happens outside this package, by
calling one stage method at a time instead of `run_all()`.

## `ProjectConfig`

One dataclass, grouped by stage. Every field's default means "use the
automatic/library behavior" -- a config supplies only what it needs to
override.

```python
from orchestration import ProjectConfig, PipelineRun

config = ProjectConfig(path="scan.h5", format="hdf5")
result = PipelineRun(config).run_all()
```

| Field | Meaning |
|---|---|
| `project_id` | Identifier used for logging and export-path templates. Default: stem of `path`. |
| `path`, `format`, `storage_mode`, `plugin_name` | Import. `plugin_name` disambiguates when more than one plugin could match. |
| `completion_answers` | Values for fields `completion.run()`'s automatic pass can't resolve (e.g. `geometry_type`, `sample_name`) -- fed through `completion.provide()`. |
| `rows`, `reference_reducer`, `dark_offset`, `projections_normalized`, `zinger_removal`, `zinger_threshold`, `zinger_size`, `ring_correction`, `ring_method` | Preprocessing. `rows=None` means every row. `projections_normalized=True` skips the flat/dark division; the projections are only scaled by their air level, and flats and darks are not needed. |
| `phase_retrieval`, `phase_method`, `phase_delta_beta`, `phase_distance_mm`, `phase_energy_kev`, `phase_pixel_um`, `phase_options` | Phase retrieval (propagation-based phase contrast), between zinger and ring removal. Off by default; `phase_delta_beta` has no default, the other numbers fall back to the `propagation_distance`, `energy` and a real `voxel_size` of the project. See doc/PHASE_RETRIEVAL.md. |
| `beam_hardening`, `bh_coefficients`, `bh_options` | Beam hardening correction (polynomial linearization, `c1, c2, c3, ...`), the last preprocessing step. Off by default; the coefficients have no default. See doc/BEAM_HARDENING.md. |
| `calibration_backend`, `center`, `tilt_deg`, `tilt_num_rows`, `tilt_margin`, `tilt_outlier_tolerance` | `center`/`tilt_deg=None` means auto-find (`find_center()`/`find_tilt()`); a value skips finding entirely. For scans longer than 180 degrees, finding uses only the first 180 degrees. |
| `reconstruction_backend`, `device`, `reconstruction_options` | Reconstruction. `reconstruction_backend` unset (the default) means the default for the project's geometry, `gridrec` or `astra_fdk` (cone), settled by `complete()`. `geometry_type` and the cone-beam/pixel-size fields are **not** here -- they come from `project.metadata`, already resolved by completion; see "Geometry fields" below. |
| `reconstruction_filter` | Reconstruction filter (`ramp`, `hann`, `hamming`, ...; INI key `filter`). `None` (default) = the backend's default. See doc/RECONSTRUCTION.md's "Filters" section. |
| `rotate_deg` | Rotate the reconstructed volume by this many degrees about the scan axis. `None` (default) = no rotation. See doc/RECONSTRUCTION.md's "Rotation" section. |
| `crop`, `dtype`, `dtype_clip_min`, `dtype_clip_max` | Postprocessing. `dtype="uint16"` runs `reconstruction.to_uint16()`. |
| `export_format`, `export_path`, `export_dataset_path`, `export_extra_settings` | Export. `export_path` may be a template; see "Export path templates" below. |
| `stream`, `stream_chunk_rows` | `stream=True` reconstructs and writes one chunk of rows at a time instead of the whole volume, so a project too large for RAM as a single reconstruction can still be run -- needs `center`/`tilt_deg` (and, for `dtype="uint16"`, the clip fields above) already resolved. `stream_chunk_rows=None` sizes chunks automatically. See doc/RECONSTRUCTION.md's "Streaming" section. |

`ring_options` and `reconstruction_options` are open dicts for
backend-specific keyword arguments, reachable only by building/editing
a `ProjectConfig` in code (e.g. `dataclasses.replace(config,
reconstruction_options={...})`) -- not exposed in the INI grammar,
since they're inherently open-ended (a future backend's kwargs can't be
enumerated in advance) rather than something a template can meaningfully
pre-fill.

### Geometry fields

`geometry_type`, `source_to_detector_distance`, `source_to_object_distance`,
and `detector_pixel_size` are not `ProjectConfig` fields. They are
`core/known_fields.py` entries, resolved onto `project.metadata` by the
completion stage (from the file itself, or from `completion_answers`)
before `reconstruct()` runs, and read from there -- the same place
`angles` comes from.

## `PipelineRun`

Each stage is independently callable and stores its result on the
instance, so a caller can inspect state between stages (a GUI plotting
calibration results before committing to reconstruction) or re-call a
stage with different values (approving or overriding a calibration
result is just calling `.calibrate()` again).

```python
run = PipelineRun(config, interactive=False)
run.import_data()     # -> ProjectObject, on run.project
run.complete()         # resolves metadata; raises/prompts if something's still missing
run.preprocess()       # -> normalized sinograms, on run.normalized
run.calibrate()        # writes center_of_rotation/tilt_angle onto run.project.metadata
run.reconstruct()      # -> volume, on run.volume
run.postprocess()      # dtype conversion / crop, in place on run.volume
run.export()           # -> the path actually written
```

`run.settings` accumulates provenance as each stage runs (backend,
center, tilt, crop, dtype clip); `.export()` merges it with
`project.export_metadata()` and writes the result as `save_volume()`'s
`settings` (see `RECONSTRUCTION.md`).

`run_all()` chains all seven and returns a `PipelineResult`
(`project_id`, `config`, `success`, `volume_path`, `error`, `log`)
instead of raising -- the shape `run_batch()` needs to keep going past
one project's failure.

## Export path templates

`export_path` can reference anything in `run.settings` (backend,
center, tilt_deg, crop, dtype_clip, ...) plus `{project_id}`, resolved
only once those stages have actually run:

```python
config.export_path = "out/{project_id}_{backend}_cor{center:.1f}.h5"
```

A placeholder with no value resolves to the literal string `"NAN"`
rather than raising -- a naming mistake shouldn't be able to fail an
export after reconstruction already ran. If the resolved path already
exists, `_1`, `_2`, ... is appended until it doesn't, so two projects
that both resolve to the same name never silently overwrite each
other.

## Config files

`orchestration/config_ini.py` reads/writes `ProjectConfig` as INI
text: plain `key = value` lines, `#` comments, no brackets or commas to
get wrong, meant to be edited by hand.

```python
from orchestration import load_config, save_config, write_config_template

write_config_template("scan.ini")   # a commented, fillable example
config = load_config("scan.ini")
save_config(config, "scan.ini")     # the reverse -- a config's current values, as single-project INI text
```

A blank value means "use `ProjectConfig`'s own default" -- the same as
not writing that line at all. `save_config()` omits a field entirely
when it's `None`, for the same reason. It round-trips every field
`FIELD_SPECS` covers, plus `completion_answers` (its own
`[completion]` section); `reconstruction_options`/`ring_options` are
never written -- see below, they're not in the INI grammar at all.

### Single-project grammar

Sections are bare stage names:

```ini
[import]
path = data/scanA.h5
format = hdf5

[completion]
geometry_type = parallel

[preprocessing]
projections_normalized = false

[calibration]
center =
tilt_deg =

[reconstruction]
backend = algotom_fbp
rotate_deg =

[export]
format = hdf5
path = out/{project_id}_volume.h5
```

The `[completion]` section isn't field-mapped like the others: every
key in it becomes one `completion_answers` entry directly, since it
corresponds to `core/known_fields.py`'s "ask" fields rather than a
fixed set of pipeline settings.

### Batch grammar

Auto-detected: any section name containing `:` switches `load_config()`
into batch mode, returning `{project_id: ProjectConfig}` instead of one
`ProjectConfig`. Sections are `[project_id:stage]`, plus an optional
`[DEFAULT:stage]` per stage that every project inherits from unless its
own section overrides a key:

```ini
[DEFAULT:reconstruction]
backend = algotom_fbp

[DEFAULT:export]
format = hdf5
path = out/{project_id}_volume.h5

[scanA:import]
path = data/scanA.h5

[scanA:calibration]
center = 56.0

[scanB:import]
path = data/scanB.h5
```

`project_id` defaults to the section token (`scanA`, `scanB`) when not
set explicitly inside that project's own section.

`[DEFAULT:stage]` is this project's own convention, not `configparser`'s
built-in `[DEFAULT]` -- that merges by key name into every section
regardless of stage, and `path` means something different in `[import]`
than in `[export]`; scoping defaults per-stage avoids that collision.

### `resolve_config()`

```python
config = resolve_config(data_path, explicit_path=None)
```

Precedence: `explicit_path` if given; otherwise a sidecar
`<data_path>.tomo_config.ini` next to the data, if one exists;
otherwise `None` -- the caller decides whether to prompt interactively
or treat that as a validation failure.

## Batch runs

```python
from orchestration import run_batch

report = run_batch([config_a, config_b, config_c])
report.succeeded   # list[PipelineResult]
report.failed      # list[PipelineResult]
report.validation_problems   # {project_id: [problem, ...]}
```

`validate_config()` dry-runs import, completion, preprocessing, and
calibration for one config -- the stages most likely to reveal a real
config/data problem -- without reconstructing or exporting
(reconstruction is the dominant cost in a real run, and a systemic
reconstruction problem surfaces immediately on whichever project runs
first anyway).

Validation is advisory only: `run_batch()` runs every config through
`run_all()` regardless of what `validate_config()` found. A project
that would have failed validation just fails again, at the same stage,
reported the same way (a failed `PipelineResult`).

Every project gets an independent attempt: one project's failure never
stops the rest, whether it's a `PipelineStageError` (`run_all()`
catches this itself) or anything else (`run_batch()` wraps each
project's `run_all()` call in a broad exception catch). `PipelineRun.run_all()`
called directly, outside `run_batch()`, does not have this broad
catch.

`PipelineResult.config` is the config that produced it, so re-running
only what failed after fixing a config is direct:

```python
fixed = [dataclasses.replace(r.config, completion_answers={"geometry_type": "parallel"}) for r in report.failed]
run_batch(fixed)
```

Batch runs are always sequential.

## Command line

`pip install -e .` registers a `tomo` command (`orchestration/cli.py`):
`tomo run`/`tomo batch`/`tomo validate`/`tomo init-config`, a thin
wrapper over the functions above. See [`CLI.md`](./CLI.md) for the full
reference.

## See also

- [`CLI.md`](./CLI.md): the `tomo` command in full -- every subcommand,
  exit codes, example output.
- [`demo/05_orchestration.ipynb`](./demo/05_orchestration.ipynb): a single
  project, `run_all()` and step-by-step, HDF5 and TIFF.
- [`demo/06_batch_orchestration.ipynb`](./demo/06_batch_orchestration.ipynb):
  several projects from one batch config file, validation, and one
  project failing without stopping the rest.
- [`RECONSTRUCTION.md`](./RECONSTRUCTION.md): `reconstruct()`,
  `to_uint16()`, `save_volume()`.
- [`CALIBRATION.md`](./CALIBRATION.md): `find_center()`, `find_tilt()`,
  `centers_for_rows()`.
- [`USING_THE_PIPELINE.md`](./USING_THE_PIPELINE.md): calling each
  pipeline stage directly, without `orchestration/`.
- [`ARCHITECTURE.md`](./ARCHITECTURE.md): how this fits with the rest
  of the codebase.
