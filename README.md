# tomo_project — skeleton (v2: simplified structure)

## Layout

```
core/            shared data model, imported by everything, imports nothing from data_import
  metadata.py    FieldSpec, MetadataBundle -- the four-tier metadata model
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
    generic.py     low-specificity fallback plugins (used when no real vendor matches)
    synthetic.py   reference plugins matching the synthetic test fixture -- COPY THIS
                    file as your template when adding a real vendor plugin

tests/
  conftest.py                     shared fixtures, incl. `synthetic_data`
  fixtures/generate_synthetic.py  generates the sphere+box+projection test data
  test_project.py                 ProjectObject / MetadataBundle logic, no files involved
  test_detection.py               plugin detection: match / fallback / no-match / ambiguous
  test_importer_end_to_end.py     full import round-trips, both formats, both storage modes
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

## Adding support for a new instrument

Copy `data_import/vendors/synthetic.py`, rename the class and its
`signature` to match your vendor's actual file layout, adjust
`get_path_map()` / `parse_metadata()`, add an import line for it in
`data_import/vendors/__init__.py`. Nothing else in the codebase needs
to change.

## Notes / what's unverified here

`h5py` and `pytest` aren't installed in the sandbox this was built in,
and there's no network access there to install them, so:
- All files were syntax-checked (`py_compile`) and reviewed, but the
  HDF5 code path itself was NOT executed here.
- The TIFF path (detection, import, metadata tiers, DataInterface
  round-trip, no-match and fallback behaviour) WAS fully exercised
  manually and passes.

First thing to do in the Claude Code / VSCode session:
`pip install -r requirements.txt && pytest tests/` to confirm the HDF5
tests too (`test_import_hdf5_ram`, `test_import_hdf5_disk`,
`test_import_hdf5_disk_materialize_to_ram`, and the hdf5 cases in
`test_detection.py`).
