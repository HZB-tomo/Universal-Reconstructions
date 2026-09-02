# Using the pipeline

This is for someone *using* the software to load and prepare a dataset —
in a notebook or a script — not someone adding support for a new
instrument. If you're writing a vendor plugin instead, see
[`WRITING_A_PLUGIN.md`](./WRITING_A_PLUGIN.md). If you want to know why
things are built the way they are, see [`ARCHITECTURE.md`](./ARCHITECTURE.md).

The pipeline, as far as it's built today, has three stages:

```
import  →  completion  →  preprocessing  →  (calibration — not built yet)
```

This document walks through all three, in order.

## 1. Importing a project

You give `DataImporter` a file (or folder, for tiff) and tell it what
format it is. It figures out the rest on its own — which instrument
wrote the file, where the data lives inside it, what metadata it
contains — you don't need to know or specify any of that.

```python
from data_import.importer import DataImporter

importer = DataImporter()
project = importer.import_project(
    "my_scan.h5",
    format="hdf5",          # or "tiff", for a folder of tiff files
    storage_mode="ram",     # or "disk" -- see below
)
print(project)
```

**`format`** is `"hdf5"` or `"tiff"` — you have to say which, the
software won't guess this part for you from the file extension.

**`storage_mode`** controls whether the projection/flat/dark arrays are
read fully into memory now (`"ram"`) or left on disk and read lazily as
you actually use them (`"disk"`). `"disk"` only works for hdf5 — tiff
data is always loaded fully into RAM, so passing `storage_mode="disk"`
for a tiff project raises a `ValueError`. Use `"disk"` for datasets too
large to comfortably fit in memory; you can always switch to RAM later
(see below).

### If the file isn't recognized

Two errors can come out of `import_project()` before you even get a
project back:

- **`NoVendorMatchError`** — nothing recognized this file at all. The
  error message lists, for every known vendor, exactly which check
  failed — useful for figuring out whether the file is genuinely
  unsupported or just doesn't quite match what's expected.
- **`AmbiguousVendorMatchError`** — more than one vendor convention
  matched equally well, so the software refuses to guess. You can
  resolve this by passing a specific plugin instance yourself via
  `import_project(..., plugin=<instance>)`, if you know which one is
  correct.

Neither of these should come up often in normal use — they mean the
file genuinely doesn't match anything the software currently knows how
to read.

### Reading the actual data

Once you have a `project`, read data and metadata out of it through
`DataInterface` — never touch `project.data_handles` directly, since
that's what hides whether you're RAM- or disk-backed:

```python
from core.data_interface import DataInterface

di = DataInterface(project)

di.get_projections()          # the full projection stack
di.get_projections(index=0)   # just one angle, without loading the rest
di.get_flats()                # None if this project has no flats
di.get_metadata("angles")
```

If you imported with `storage_mode="disk"` and later decide you want
everything in memory, `di.materialize_to_ram()` converts every handle
over — one-directional, explicit, does nothing automatically.

## 2. Completion — filling in what the file didn't tell you

A vendor's file rarely contains everything the software needs. Some of
that is recoverable automatically (a default, or working it out from
other values already present); the rest genuinely has to come from you.
That's what the completion stage is for.

### Checking what's there

```python
from core.known_fields import status

for row in status(project.metadata):
    print(row.key, row.tier, row.resolved, row.value)
```

This reports on every field the software actually depends on — not just
whatever the file happened to mention — so it also tells you about
fields the file is silently missing entirely. See
[`PROJECT_METADATA.md`](./PROJECT_METADATA.md) for what each field means,
its unit, and valid values.

### Filling in what can be filled in automatically

```python
import completion

completion.run(project)
```

This applies defaults and works out anything derivable from data
already present — e.g. `angles` from `angle_span` and the projection
count, or vice versa, whichever one the vendor didn't already provide.
It's safe to call as often as you like — anything already resolved is
left alone. Some fields (`geometry_type`, for instance) have no way to
be worked out automatically at all — there's no safe default and
nothing to derive it from — so they'll still be missing after this,
waiting on you (see the note at the end of this section for the full
list of what completion can and can't do on its own).

### Checking what's still needed

```python
project.needs_user_input()      # True/False
project.missing_soft_fields()   # e.g. ["geometry_type", "angle_span"]
```

### Supplying the answers yourself

```python
completion.provide(project, "geometry_type", "cone")
completion.provide(project, "angle_span", 360.0)
```

Each call records your answer and immediately re-runs the automatic
pass — so one answer can unlock others. For example, answering
`geometry_type="cone"` will make cone-beam-only fields like
`source_to_detector_distance` show up as newly required, where they
were correctly hidden before (a parallel-beam project never asks for
them at all).

A typical loop looks like this:

```python
completion.run(project)
while project.needs_user_input():
    for key in project.missing_soft_fields():
        value = ...  # however you're collecting it -- input(), a config file, a form
        completion.provide(project, key, value)
```

### What completion does *not* do yet

`angles`, `angle_span`, and `detector_pixel_size` all have real derive
logic now (`completion/derivers.py`) — but a deriver only helps if its
*own* inputs are available. `angles` derives from `angle_span` (plus the
projection count and `rotation_direction`); if a vendor provides
neither `angles` nor `angle_span`, there's nothing to derive either one
from, and both stay missing until you supply at least one via
`completion.provide()`. There's also no calculation of any kind for
`center_of_rotation` or `tilt_angle` — those are `computed`-tier and
deliberately untouched by completion; they're the job of the
calibration stage, which comes after preprocessing (below) and doesn't
exist yet.

## 3. Checking completion readiness

```python
project.needs_user_input()   # False once completion is done
```

Once this is `False`, every field the software depends on for the next
stage is either filled in or was never required in the first place —
that's what "completion-ready" means. `status(project.metadata)` still
gives you the full picture if you want to see it directly rather than
just the yes/no answer.

## 4. Preprocessing — normalization and artifact removal

Once a project is completion-ready, `preprocessing/` turns its raw
projections into reconstruction-ready sinograms: flat/dark correction
first, then (optionally) zinger and ring-artifact removal. These
functions read a `ProjectObject`'s raw data but don't need its metadata
to be fully resolved — flat/dark correction only cares about the
`projections`/`flats`/`darks` arrays — so technically nothing stops you
from calling this stage before completion finishes. In practice you
still want completion done first: calibration (next, unbuilt) needs
`geometry_type`/`angle_span` to pick the right approach, so there's no
reason to reorder the stages just because one function happens not to
require it.

### Flat/dark correction

```python
from preprocessing import build_reference, normalize, normalize_slice, normalize_rows

reference = build_reference(project)              # reduces flats/darks once -- reuse this
normalize(raw_array, reference)                    # the raw formula, if you already have an array
normalize_slice(project, row=64, reference=reference)          # one sinogram, all angles
normalize_rows(project, slice(100, 200), reference=reference)  # a batch of rows
normalize_rows(project, slice(None), reference=reference)      # the entire volume
```

`build_reference()` raises `MissingFlatFieldError` if the project has
no flat field data at all — there's no fallback for that. A missing
dark field is fine: `reference.dark` becomes a scalar `0.0` instead of
a 2D image, and the same formula works either way via plain numpy
broadcasting.

`normalize_slice()`/`normalize_rows()`/the whole-volume call all
produce identical results for the same rows — they're the same
underlying formula, just handed different-shaped input. Pick whichever
matches what you're doing: a single test slice while exploring, a
bounded batch to keep memory in check on a large disk-backed project,
or the whole volume at once if you have the RAM for it.

Deliberately **not** included here: `-log()`. Every real pipeline this
project's `oldscripts/` were checked against applies that right
before/inside reconstruction, never as part of flat/dark correction —
ring removal and (later) reconstruction itself both expect the plain
transmission data these functions produce.

### Zinger and ring-artifact removal (optional)

```python
from preprocessing import remove_zinger, remove_ring_artifacts, process_volume

clean = remove_zinger(sinogram, threshold=0.08)
clean = remove_ring_artifacts(sinogram, method="sorting")   # or "filtering"/"fitting"/"normalization"/"all"

# batched across every row of a volume, in parallel:
clean_volume = process_volume(volume, remove_ring_artifacts, method="sorting")
```

Run these on `normalize()`'s output, in that order (matches
`oldscripts/full_reconstruction_cli.py`'s pipeline exactly: flat/dark
correction → zinger removal → ring removal → reconstruction).

These need the `algotom` package, which is **not** installed by
default — `pip install .[preprocessing]` (or `pip install algotom`
directly). Nothing else in this project needs it. Call one of these
three functions without it installed and you get a plain `ImportError`
telling you what to install, rather than this whole package failing to
import — `import preprocessing` itself always works regardless.

### Where does the result go?

Nowhere, on purpose. Every function in `preprocessing/` is pure: array
in, array out. Nothing gets written back onto `project` or
`project.data_handles` — there's no `project.normalized_projections`
to find later. Two reasons this is deliberate, not unfinished:

- A large in-memory result attached to `project` lives exactly as long
  as `project` does. A pipeline script that uses a batch and lets it go
  out of scope avoids holding onto memory it doesn't need anymore;
  attaching results to a long-lived object would work against that.
- With a real pipeline script driving the sequence (import → complete →
  preprocess → calibrate → reconstruct, in a straight line), there's no
  ambiguity about "has this been normalized yet" for the script to
  resolve — the script's own control flow already answers that. A
  stored flag would be solving a problem that only shows up once
  something *other* than one linear script needs to ask the question —
  a GUI with multiple independent entry points, for instance. Worth
  revisiting then, not now.

## Full example

```python
from data_import.importer import DataImporter
from core.known_fields import status
import completion
from preprocessing import build_reference, normalize_rows

importer = DataImporter()
project = importer.import_project("my_scan.h5", format="hdf5", storage_mode="ram")

completion.run(project)
while project.needs_user_input():
    for key in project.missing_soft_fields():
        print(f"{key} is missing -- ", end="")
        value = input(f"enter a value for {key}: ")
        completion.provide(project, key, value)

assert not project.needs_user_input()
print("Project is completion-ready:")
for row in status(project.metadata):
    print(f"  {row.key}: {row.value!r}")

reference = build_reference(project)
normalized = normalize_rows(project, slice(None), reference)
# ... hand `normalized` off to calibration / reconstruction, once they exist
```

## Where to go next

- [`doc/demo/02_completion.ipynb`](./demo/02_completion.ipynb) — the
  completion workflow above, run end-to-end against a generated example
  dataset with real output.
- [`PROJECT_METADATA.md`](./PROJECT_METADATA.md) — the full field
  reference: every field's tier, resolution strategy, unit, and notes.
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — how and why the codebase is
  structured this way.
