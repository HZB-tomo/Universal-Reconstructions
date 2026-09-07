# Architecture

This document explains how the codebase is organized, why it's organized
that way, and how data flows through it. If you're new to this project,
read this top to bottom once — everything else (docstrings in the code,
the other docs in this folder) will make more sense afterward.

## The big picture

Six packages so far:

```
core/            the shared data model — every other package depends on this
data_import/     turns a file on disk into a ProjectObject
completion/      fills in metadata a vendor's file didn't provide
preprocessing/   turns raw projections into reconstruction-ready sinograms
calibration/     finds center-of-rotation and rotation-axis tilt
reconstruction/  turns sinograms into reconstructed images
```

`core/` is the foundation. It defines what a "loaded dataset" *is*
(`ProjectObject`), how you read data out of one (`DataInterface`), how
that data is actually stored (`DataHandle`), how metadata is tracked
(`MetadataBundle`), and — since `completion/` was added — the single
catalog of every metadata field the software actually depends on
(`known_fields.py`). Nothing in `core/` knows anything about files,
hdf5, tiff, or vendors — it would work identically if data arrived by
some other means entirely.

`data_import/` is the one and only thing that *produces* a
`ProjectObject`. It knows how to open hdf5/tiff files, how to figure out
which vendor wrote a given file, and how to read that vendor's
metadata into the shape `core/` expects.

`completion/`, `preprocessing/`, `calibration/`, and `reconstruction/`
are the pipeline stages after import — see
[`USING_THE_PIPELINE.md`](./USING_THE_PIPELINE.md) for the end-user
walkthrough of all four. `completion/` and `preprocessing/` depend only
on `core/`, exactly as originally planned here: neither imports
anything from `data_import/`, they just take a `ProjectObject` that
already exists and work with it via `DataInterface`. The one-way
dependency rule from the original plan held up in practice.

`calibration/` and `reconstruction/` are the exception, deliberately:
neither depends on `core/` at all, or takes a `ProjectObject` — see
`reconstruction/`'s own section below for why that decoupling was a
real design correction, not the original plan; `calibration/` was built
the same way from the start, for the same reason (see
[`CALIBRATION.md`](./CALIBRATION.md)).

## `core/` in detail

### `MetadataBundle` (`core/metadata.py`)

Every piece of metadata about a dataset — voxel size, angles, center of
rotation, operator name, whatever — is stored as a `FieldSpec`: a value,
plus which of four **tiers** it belongs to:

| Tier | Meaning | Example |
|---|---|---|
| `mandatory_hard` | Missing → import fails outright | projections |
| `mandatory_soft` | Missing → import succeeds, but the project is flagged as needing user input before certain later stages can run | voxel size, geometry type |
| `optional` | Nice to have, no consequence if missing | operator name |
| `computed` | Usually filled in later by an algorithm; may start as a rough guess from the file | center of rotation |

A `MetadataBundle` doesn't know anything about hdf5 or tiff — vendor
plugins (see below) are the ones who decide which tier each field
belongs to for their particular vendor, based on what that vendor
typically does or doesn't provide.

Two questions the rest of the software actually needs to ask are
answered directly by this class:
- `missing_hard()` — is this dataset even usable?
- `missing_soft()` — does the user need to be prompted for anything?

Note: `computed` fields are **overwritten, not versioned**. If an
algorithm calculates a new center of rotation, the old value (whether it
came from a file or a previous calculation) is simply replaced. We
decided against keeping a history log for this, to keep the model
simple — this can be revisited later if a real need for it shows up.

### `DataHandle` (`core/handles.py`)

The actual array data (projections, flats, darks) is never stored
directly on a `ProjectObject` as a raw numpy array. Instead it's wrapped
in a `DataHandle`, which is either:

- **`RamHandle`** — a plain numpy array already in memory.
- **`Hdf5DiskHandle`** — a reference to a *still-open* `h5py.Dataset`,
  which reads from disk only when something actually asks for data.

Both expose the exact same interface (`.get()`, `.shape`,
`.storage_mode`), so code that reads data doesn't need to know or care
which kind it has. This is also why `ProjectObject` stays cheap to
create and inspect even for huge datasets — it never holds the actual
array, just a handle pointing at where the data lives.

TIFF data is **always** a `RamHandle` — per an early project decision,
TIFF folders are always fully loaded into memory at import time, since
there's no lazy-read equivalent for a folder of individual image files
the way there is for hdf5 datasets.

### `ProjectObject` (`core/project.py`)

The single object that represents "one loaded dataset" as it moves
through the software. Deliberately passive — it holds:

- `format` — `"hdf5"` or `"tiff"`
- `storage_mode` — `"ram"` or `"disk"`
- `metadata` — a `MetadataBundle`
- `data_handles` — a dict of `DataHandle`s, keyed `"projections"`,
  `"flats"`, `"darks"`
- `vendor_plugin_name` — which plugin produced this project, for
  debugging/logging
- `source_info` — free-form dict with things like the original file
  path and the `PathMap` used (see below)

It also exposes two readiness checks that any later stage can call:
`is_ready_for_import()` and `needs_user_input()`.

`ProjectObject` is created by `DataImporter` and never by anything else.

### `DataInterface` (`core/data_interface.py`)

The facade every other part of the software should use to actually get
data out of a `ProjectObject` — `get_projections()`, `get_flats()`,
`get_darks()`, `get_metadata()`, `set_computed()`. It exists purely to
hide the RAM-vs-disk distinction: call `get_projections()` on a
disk-backed or RAM-backed project and you get a numpy array back either
way.

`DataInterface` also has `materialize_to_ram()`, which converts every
disk-backed handle in a project into a RAM handle — the explicit,
one-directional way a user says "load everything into memory now."

### `known_fields.py` (`core/known_fields.py`)

The single source of truth for every metadata field the software
depends on: its tier (from `metadata.py`), its unit, and its
**resolution strategy** — `derive` (compute it from other known
fields), `default` (a fixed fallback), `ask` (no safe default, needs a
human), or `calculate` (an algorithm proposes a value, a human
confirms — calibration's job, not completion's). `MetadataBundle`
itself stays deliberately generic and doesn't know what `voxel_size`
means; this file is what stops that genericity from meaning every
vendor plugin has to redeclare a field's tier itself, with the drift
risk that implies.

Two functions matter here: `reconcile(bundle)` fills in any known field
a plugin never mentioned at all (declared missing, with the registry's
tier) — called from `data_import`'s `import_project()` right after a
plugin's `parse_metadata()` returns, so `mandatory_hard` enforcement no
longer depends on every plugin author remembering to self-report a
missing field correctly. `status(bundle)` gives a full readout of every
known field's current state — see `completion/` below for what
consumes it. See [`PROJECT_METADATA.md`](./PROJECT_METADATA.md) for the
human-readable field-by-field reference.

## `data_import/` in detail

This package answers one question: given a file path, produce a ready
`ProjectObject`. The work is split into three concerns that never mix:

```
"How do I recognize this vendor's file?"    → signature.py (a plugin's `signature`)
"Where does this vendor put things?"        → a plugin's get_path_map() / parse_metadata()
"How do I physically read hdf5/tiff?"       → backend.py
```

### Signatures and detection (`data_import/signature.py`)

Detection is **deterministic**, not a confidence score. A `Signature` is
just a checklist of conditions (required hdf5 paths, required attribute
values, or for tiff, filename patterns and sidecar file requirements).
Every condition either holds or doesn't — there's no fuzzy scoring to
interpret.

The number of conditions a signature checks is its **specificity**.
When `DataImporter` is trying to figure out which plugin should handle
a file:

1. Every registered plugin's signature is checked against the file.
2. If none match → `NoVendorMatchError`, with the exact failed
   conditions for every plugin, so you can see precisely why.
3. Among plugins that *do* match, the one with the **highest
   specificity** wins — it "knows more" about this exact file
   structure, so it's less likely to be a coincidental match.
4. If two or more plugins are tied for the highest specificity →
   `AmbiguousVendorMatchError`. We never silently guess.

This is why the built-in fallback plugins (see below) use very low
specificity: they only ever win when nothing more specific matches.

### The plugin contract (`data_import/plugin.py`)

Every vendor plugin subclasses `ImportPlugin` and implements exactly
three things:

- `signature` — a class attribute, `Hdf5Signature` or `TiffSignature`
- `get_path_map(handle)` — given the open file/folder, return a
  `PathMap` pointing to where projections/flats/darks/angles live
- `parse_metadata(handle, path_map)` — read metadata values and sort
  them into the four tiers, returning a `MetadataBundle`

A plugin never opens a file itself (`backend.py`'s job) and never
decides RAM vs disk (the caller's choice, passed through
`DataImporter`). A plugin is pure knowledge about one vendor's
conventions.

Plugins register themselves with `@register_plugin` when their module
is imported — see `data_import/vendors/__init__.py`, which imports
every vendor module so this happens automatically on startup. The
registry itself (`get_plugins()`, `clear_registry()`) is just a plain
in-memory list per format.

**Important for future work**: this plugin pattern is intentionally
scoped to `data_import/` only. It's not a generic pattern pre-built for
other future modules (reconstruction, viewing, GUI). Reach for a
plugin/registry pattern again only if a future module genuinely needs
"multiple interchangeable implementations, auto-selected" the way
vendor import does — most modules won't.

### Format backends (`data_import/backend.py`)

`Hdf5Backend` and `TiffBackend` know how to physically read their
format, with zero knowledge of vendor conventions. The same backend is
reused by every plugin of that format — this is also what makes it
possible to later reuse `Hdf5Backend` for *writing* data (e.g.
exporting a reconstructed volume), without needing per-vendor export
code.

### `DataImporter` (`data_import/importer.py`)

The orchestrator. `detect()` figures out which plugin should handle a
file, without loading data. `import_project()` runs the full pipeline:

1. Open the file/folder via the right backend.
2. Detect (or use a caller-supplied) plugin.
3. Ask the plugin where things are (`get_path_map`) and what the
   metadata means (`parse_metadata`).
4. If any `mandatory_hard` field is missing → `ProjectImportError`.
5. Ask the backend to actually read the arrays at the paths the plugin
   pointed to, building `DataHandle`s in the requested storage mode.
6. Assemble and return the `ProjectObject`.

### `vendors/` — one file per vendor

Each file here is either a real vendor plugin, or (for `generic.py`)
the low-specificity fallback pair that catches anything no real plugin
recognizes. See `WRITING_A_PLUGIN.md` for how to add a new one.

## `completion/` in detail

Given a `ProjectObject` fresh out of `DataImporter`, drive every
`mandatory_soft` field toward resolved, using whichever strategy
`core/known_fields.py` says applies. Two entry points:

- **`run(project)`** — the automatic pass: tries `derive` then
  `default`, in order, for every currently-missing field, and repeats
  until a full pass makes no further progress (fields can depend on
  each other — e.g. `angles` needs `rotation_direction`, which only
  gets its own default applied during the same pass — so a single pass
  isn't always enough). Never attempts `ask` or `calculate` — those
  need a human, which is exactly what this function doesn't provide.
- **`provide(project, key, value)`** — records an externally-collected
  answer (`source="user"`), then re-runs the automatic pass, since one
  answer can unblock others.

Deliberately does no asking itself — there's no GUI yet (today:
notebook/scripting; later: a GUI), and a blocking prompt baked into
this code would only work for whichever one happens to be driving it.
`core/known_fields.status(project.metadata)` is how a caller sees
what's left.

The actual per-field derive logic (`completion/derivers.py`) is kept
separate from the generic pass/loop logic (`completion/engine.py`), the
same split as `data_import/plugin.py` (the contract) vs.
`data_import/vendors/*.py` (the actual per-vendor knowledge). Built-in
derivers: `angle_span` (from `angles`), `angles` (from `angle_span` +
projection count + `rotation_direction`), `detector_pixel_size` (from
`voxel_size`). New derivers register via `@register_deriver("field_name")`,
mirroring `@register_plugin`.

`completion/` never touches `computed`-tier fields (`center_of_rotation`,
`tilt_angle`) — those are `calibration/`'s job (see below), not something
completion derives or asks the user for.

## `preprocessing/` in detail

Turns a completion-ready project's raw projections into
reconstruction-ready sinograms. Two files, two different dependency
postures:

- **`normalization.py`** — flat/dark correction,
  `(raw - dark) / (flat - dark)`. Pure numpy, no optional dependency.
  `build_reference(project)` reduces the flats/darks stacks to one
  reference image (or, for dark, a scalar offset if the project has no
  dark field — flats have no such fallback and raise
  `MissingFlatFieldError` if absent) — computed once, reused across
  calls. `normalize()` is pure broadcasting: the same function handles
  a single sinogram, a batch of rows, or the whole volume, purely by
  what shape of array it's given; `normalize_slice()`/`normalize_rows()`
  are `DataInterface`-based convenience wrappers that handle the
  row-slicing bookkeeping for whichever granularity you're working at.
  Deliberately does **not** apply `-log()` — every real pipeline in
  `oldscripts/` applies that right before/inside reconstruction, never
  as part of flat/dark correction.
- **`corrections.py`** — zinger removal and ring-artifact removal
  (four interchangeable algorithms — sorting/filtering/fitting/
  normalization — plus a combined "all" method), both backed by the
  `algotom` library. Unlike `normalize()`, these can't be vectorized
  into one broadcastable op (sorting/filtering/fitting a sinogram is
  inherently a per-2D-array algorithm), so batching means running the
  same single-sinogram function once per row in parallel —
  `process_volume()` delegates that to algotom's own
  `parallel_process_slices()`. `algotom` is deliberately **not** a
  mandatory dependency of this project (see `pyproject.toml`'s
  `preprocessing` extra) — every function here imports it lazily and
  raises a plain, actionable `ImportError` if it's missing, rather than
  this whole module failing to import for everyone.

`preprocessing/`'s functions are pure: array in, array out. They don't
write anything back onto `ProjectObject`, and deliberately so — see
`USING_THE_PIPELINE.md` for why that's the current design, not an
oversight.

## `calibration/` in detail

Finds center-of-rotation and rotation-axis tilt from normalized
projections. Structured exactly like `reconstruction/` (below), on
purpose: `backend.py` (`CalibrationBackend`, one method —
`find_center()` — plus the same `register_backend`/`get_backend`
registry idiom), `backends/algotom_cor.py` (the one built-in backend),
`api.py` (`find_center()`/`find_tilt_and_center()`, the backend-agnostic
orchestration). If `algotom` is ever unavailable, a from-scratch
reimplementation of the published algorithm it wraps only has to
satisfy that one method to slot in — see
[`CALIBRATION.md`](./CALIBRATION.md).

Doesn't take a `ProjectObject` either, for the same reason as
`reconstruction/`: the caller prepares whatever normalized data it
needs beforehand. `find_tilt_and_center()` never reconstructs anything
— it's pure, deterministic sinogram-domain math (independent
per-row `find_center()` calls plus a line fit) — which turned out to
matter a lot in practice: `reconstruction.reconstruct()`'s CPU-fallback
FBP path was verified to NOT be bit-reproducible between identical
calls, which rules out any calibration design based on reconstructing
candidates and scoring them automatically. See `CALIBRATION.md`'s
"Why this method, and not a reconstruction-quality search" section for
the full story, including what *is* still a good use for
`reconstruct()` here (a manual/visual center check with no automatic
scoring at all).

## `reconstruction/` in detail

Turns normalized sinograms into reconstructed images. Doesn't take a
`ProjectObject` — `reconstruct()` takes plain arrays, prepared by the
caller beforehand, the same way a caller already calls
`preprocessing.normalize_rows()` itself. This still matters for a
manual/visual calibration check (reconstruct several candidate centers
at once — `reconstruct()`'s stack-plus-per-slice-center support handles
this natively — and look at them), even though the automatic
`calibration/` module ended up not needing it (see above). See
[`RECONSTRUCTION.md`](./RECONSTRUCTION.md) for the full reasoning.

- **`backend.py`** — `Geometry` (parallel/cone, field names matching
  `core/known_fields.py`) and `ReconstructionBackend`, the plugin
  contract: one method, `reconstruct(sinograms, angles, center, geometry,
  device, **options)`. Registered via `@register_backend`/looked up via
  `get_backend(name)` — the same idiom as `@register_plugin`, but
  deliberately without a `Signature`/`matches()` equivalent, since
  there's nothing in the data to auto-detect a reconstruction backend
  from — selection is always explicit.
- **`api.py`** — the public `reconstruct()`. One function, not two:
  a single 2D sinogram in gives a single 2D image back, a 3D stack in
  gives a matching 3D stack back. The stack's axis order is
  deliberate: input is `(n_angles, n_slices, width)`, matching
  `preprocessing.normalize_rows()`'s own output exactly (so its result
  can be handed straight to `reconstruct()`, no reshaping); output is
  `(n_slices, height, width)`, matching this codebase's own volume
  convention instead (e.g. the phantom generator's `(height, width, width)`)
  — different kinds of stacks, matching two different existing
  conventions on purpose.
- **`backends/algotom_fbp.py`** — the one built-in backend, parallel-beam
  only, wrapping `algotom.rec.reconstruction.fbp_reconstruction` (which
  only takes one sinogram at a time — this backend loops, unparallelized,
  the simplest correct reference implementation, not a claim of being
  the fastest one). Same optional-`algotom` pattern as
  `preprocessing/corrections.py`.

How a backend gets from a batch of sinograms to a batch of images —
loop, native batch call, CPU, GPU — is entirely that backend's own
business; the contract only specifies shapes in and out. This matters
for genuinely different backends: a CPU-bound library benefits from
parallelizing across independent slices, while a GPU-batch-native
library needs its whole batch in one call to avoid paying per-slice
transfer/launch overhead N times — see `RECONSTRUCTION.md` for the
fuller version of this reasoning.

## Data flow, end to end

```
file path
  │
  ▼
backend.open()              (data_import/backend.py)
  │
  ▼
plugin.matches()  ×N        every registered plugin checks its signature
  │
  ▼
highest specificity wins    (or NoVendorMatchError / AmbiguousVendorMatchError)
  │
  ▼
plugin.get_path_map()       where are projections/flats/darks/angles?
  │
  ▼
plugin.parse_metadata()     read metadata, sort into 4 tiers
  │
  ▼
known_fields.reconcile()    catch any known field the plugin never mentioned
  │
  ▼
backend.build_data_handle() × (projections, flats, darks)
  │
  ▼
ProjectObject assembled     ← this is what DataImporter.import_project() returns
  │
  ▼
DataInterface(project)      everything downstream reads through here
  │
  ▼
completion.run() / .provide()   drive mandatory_soft fields to resolved
  │
  ▼
preprocessing.normalize() / remove_zinger() / remove_ring_artifacts()
  │
  ▼
calibration.find_tilt_and_center()   normalized projections -> (tilt_deg, center)
  │
  ▼
reconstruction.reconstruct()    sinogram(s) -> image(s), via the chosen backend
```
