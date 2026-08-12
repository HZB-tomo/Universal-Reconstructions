# Architecture

This document explains how the codebase is organized, why it's organized
that way, and how data flows through it. If you're new to this project,
read this top to bottom once — everything else (docstrings in the code,
the other docs in this folder) will make more sense afterward.

## The big picture

There are two packages right now:

```
core/            the shared data model — every future module depends on this
data_import/     turns a file on disk into a ProjectObject
```

`core/` is the foundation. It defines what a "loaded dataset" *is*
(`ProjectObject`), how you read data out of one (`DataInterface`), how
that data is actually stored (`DataHandle`), and how metadata is tracked
(`MetadataBundle`). Nothing in `core/` knows anything about files, hdf5,
tiff, or vendors — it would work identically if data arrived by some
other means entirely.

`data_import/` is the one and only thing (so far) that *produces* a
`ProjectObject`. It knows how to open hdf5/tiff files, how to figure out
which vendor wrote a given file, and how to read that vendor's
metadata into the shape `core/` expects.

The dependency only goes one direction: `data_import` imports from
`core`, never the other way around. Future modules (reconstruction,
viewing, a GUI) will each depend on `core/` the same way, and probably
won't need to depend on `data_import/` at all — they just take a
`ProjectObject` that already exists and work with it via
`DataInterface`.

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
backend.build_data_handle() × (projections, flats, darks)
  │
  ▼
ProjectObject assembled     ← this is what DataImporter.import_project() returns
  │
  ▼
DataInterface(project)      everything downstream reads through here
```
