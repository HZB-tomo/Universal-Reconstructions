# tomo_project — skeleton (v2: simplified structure)

## Layout

```
core/            shared data model, imported by everything, imports nothing from data_import
  metadata.py    FieldSpec, MetadataBundle -- the four-tier metadata model
  known_fields.py   the catalog of every metadata field the software depends
                     on: tier + resolution strategy (derive/default/ask/calculate) + unit
  handles.py     DataHandle, RamHandle, Hdf5DiskHandle -- RAM vs disk array access
  project.py     ProjectObject -- the container that flows through the pipeline
  data_interface.py   DataInterface -- the facade for reading data out of a ProjectObject

data_import/     everything about turning a file into a ProjectObject
  path_map.py    PathMap -- where things are, once located
  signature.py   Signature/Hdf5Signature/TiffSignature -- deterministic
                 "is this my file?" checks + specificity scoring
  plugin.py      ImportPlugin (the contract) + the plugin registry
  backend.py     FormatBackend/Hdf5Backend/TiffBackend -- physical I/O
  importer.py    DataImporter -- orchestrates detect -> plugin -> backend -> ProjectObject
  errors.py      NoVendorMatchError, AmbiguousVendorMatchError, ProjectImportError
  vendors/       one file per vendor plugin
    generic.py                     low-specificity fallback plugins (no real vendor matched)
    synthetic.py                   reference plugins matching the synthetic test fixture --
                                    COPY THIS file as your template when adding a real vendor
    hypothetical_synchrotron.py    a second worked example (flats + voxel_size from file)

completion/      fills in metadata a vendor's file didn't provide -- see USING_THE_PIPELINE.md
  engine.py      run() (automatic pass) / provide() (record a user answer) / register_deriver()
  derivers.py    the built-in derive functions (angles, angle_span, detector_pixel_size)

preprocessing/   turns raw projections into reconstruction-ready sinograms
  normalization.py  flat/dark correction -- pure numpy, no optional dependency
  corrections.py    zinger/ring-artifact removal -- needs the optional `algotom` package

tests/
  conftest.py                     shared fixtures, incl. `synthetic_data`
  fixtures/                       synthetic-data generators, one per vendor convention
  test_project.py                 ProjectObject / MetadataBundle logic, no files involved
  test_detection.py               plugin detection: match / fallback / no-match / ambiguous
  test_importer_end_to_end.py     full import round-trips, both formats, both storage modes
  test_known_fields.py            reconcile() / status() against core/known_fields.py
  test_completion.py              completion.run() / .provide()
  test_derivers.py                the built-in derive functions
  test_normalization.py           flat/dark correction
  test_corrections.py             zinger/ring-artifact removal (skipped if algotom isn't installed)

doc/
  ARCHITECTURE.md          how and why the codebase is structured this way -- read this first
  USING_THE_PIPELINE.md    end-user walkthrough: import -> completion -> preprocessing
  PROJECT_METADATA.md      every metadata field's tier, strategy, unit, and notes
  WRITING_A_PLUGIN.md      how to add support for a new instrument
  GETTING_STARTED.md       environment setup
  demo/                    worked Jupyter notebooks
```

## What changed from the first draft

- Collapsed `core/`, `importer/`, `backends/`, `plugins/` (4 folders, each
  with its own `base.py`) down to just `core/` and `data_import/`. The
  concepts (backend vs plugin vs orchestrator) are still separate
  classes/files, just not separate folders -- there wasn't enough code
  yet to justify the extra navigation.
- The "plugin" pattern is now explicitly scoped to `data_import/` only.
  It's NOT a generic pattern pre-built for future modules (reconstruction,
  viewing, GUI) -- those should only reach for a plugin/registry pattern
  later if they actually need "multiple interchangeable implementations,
  auto-selected" the way vendor import does. Most won't.
- Every class and function now has a docstring explaining what it does
  AND why it exists / how it fits with the rest of the code, not just
  its signature.

## Setup

```
pip install -r requirements.txt
pytest tests/
```

See [`doc/GETTING_STARTED.md`](./doc/GETTING_STARTED.md) for the full
walkthrough (virtual env, editable install, Jupyter kernel setup, first
import). `preprocessing/corrections.py`'s zinger/ring-artifact removal
needs the optional `algotom` package — `pip install .[preprocessing]`,
or `pip install algotom` directly; nothing else in this project
requires it, and `tests/test_corrections.py` skips cleanly if it isn't
installed.

## Adding support for a new instrument

Copy `data_import/vendors/synthetic.py`, rename the class and its
`signature` to match your vendor's actual file layout, adjust
`get_path_map()` / `parse_metadata()`, add an import line for it in
`data_import/vendors/__init__.py`. Nothing else in the codebase needs
to change. See [`doc/WRITING_A_PLUGIN.md`](./doc/WRITING_A_PLUGIN.md)
for the full walkthrough.

## Where to go next

- [`doc/ARCHITECTURE.md`](./doc/ARCHITECTURE.md) — how and why the
  codebase is structured this way; read this first.
- [`doc/USING_THE_PIPELINE.md`](./doc/USING_THE_PIPELINE.md) — using
  the pipeline end to end: import, completion, preprocessing.
