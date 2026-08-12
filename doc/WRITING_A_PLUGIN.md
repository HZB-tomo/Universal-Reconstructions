# Writing a new vendor plugin

This walks through adding support for a new instrument, using the same
worked example already in the codebase (`hypothetical_synchrotron.py`)
as reference. By the end you'll have touched exactly two files, plus
optionally a test.

## What you're building

A plugin answers three questions about **one vendor's file convention**:

1. **"Is this my file?"** — a signature: a checklist of conditions
   (required hdf5 paths/attributes, or tiff filename/sidecar patterns).
2. **"If so, where's everything?"** — a `PathMap` pointing to
   projections, flats, darks, angles.
3. **"What does the metadata mean?"** — reading values and sorting them
   into the four tiers (`mandatory_hard`, `mandatory_soft`, `optional`,
   `computed`).

A plugin never opens a file itself and never decides RAM vs disk — see
`ARCHITECTURE.md` for why that separation exists.

## Step 1 — figure out the file layout

Before writing any code, know your vendor's actual convention. For
example, say a fictional instrument always writes:

```
/exchange/data          projections, 3D
/exchange/data_white    flats, 3D
/exchange/data_dark     darks, 3D
/exchange/theta         angles, 1D
root attribute: instrument_manufacturer == "Zeiss"
```

## Step 2 — create the plugin file

Create `data_import/vendors/my_vendor.py`:

```python
from data_import.plugin import ImportPlugin, register_plugin
from data_import.path_map import PathMap
from data_import.signature import Hdf5Signature
from core.metadata import MetadataBundle

_PROJECTIONS = "/exchange/data"
_FLATS = "/exchange/data_white"
_DARKS = "/exchange/data_dark"
_ANGLES = "/exchange/theta"


@register_plugin
class MyVendorHdf5Plugin(ImportPlugin):
    name = "my_vendor_hdf5"
    format = "hdf5"

    signature = Hdf5Signature(
        required_paths=[_PROJECTIONS, _FLATS, _DARKS, _ANGLES],
        required_attrs={"/": {"instrument_manufacturer": "Zeiss"}},
    )

    def get_path_map(self, handle) -> PathMap:
        return PathMap(
            projections=_PROJECTIONS,
            flats=_FLATS,
            darks=_DARKS,
            angles=_ANGLES,
        )

    def parse_metadata(self, handle, path_map: PathMap) -> MetadataBundle:
        bundle = MetadataBundle()

        bundle.set("projections_path", path_map.projections, tier="mandatory_hard")
        bundle.set("angles", handle[_ANGLES][()], tier="mandatory_hard")

        if "/exchange/pixel_size" in handle:
            bundle.set("voxel_size", handle["/exchange/pixel_size"][()], tier="mandatory_soft")
        else:
            bundle.declare_missing("voxel_size", tier="mandatory_soft")

        bundle.declare_missing("geometry_type", tier="mandatory_soft")
        bundle.declare_missing("center_of_rotation", tier="computed")

        return bundle
```

A few things worth calling out:

- **Signature specificity matters.** Here we check 4 paths + 1
  attribute = specificity 5. This needs to beat the generic fallback
  (specificity 1) and not accidentally tie with any other vendor
  plugin. More required conditions = more confidence this is really
  your vendor's file, not a coincidental match.
- **Tiering is a judgment call specific to this vendor.** Notice
  `voxel_size` is *conditionally* read here — some vendors always
  include it (in which case you'd just read it unconditionally, like
  `hypothetical_synchrotron.py` does), others sometimes omit it. Only
  the plugin author knows which applies to their vendor.
- **Don't load the big arrays here.** `parse_metadata()` should only
  read small things — scalars, short arrays like the angle list. The
  actual projections/flats/darks arrays are read later, by the backend,
  using the paths you returned from `get_path_map()`.

## Step 3 — register it

Add one import line to `data_import/vendors/__init__.py`:

```python
from data_import.vendors import my_vendor  # noqa: F401
```

That's it. `DataImporter` will now automatically try this plugin
whenever `import_project(..., format="hdf5")` is called — no other file
in the codebase needs to change.

## Step 4 (recommended) — write a test-data generator and test

Real vendor files are hard to test against without access to real
instrument data. The pattern used elsewhere in this codebase
(`tests/fixtures/generate_synthetic.py`,
`tests/fixtures/generate_hypothetical_synchrotron.py`) is to write a
small script that builds a synthetic volume, projects it, and writes it
out in your new vendor's exact layout — then test against that.

```python
# tests/fixtures/generate_my_vendor.py
import h5py
from tests.fixtures.generate_synthetic import make_volume, project_parallel

def generate(path, shape=(24, 24, 32), n_angles=10, seed=None):
    volume = make_volume(shape=shape, seed=seed)
    projections, angles = project_parallel(volume, n_angles=n_angles)
    with h5py.File(path, "w") as f:
        f.attrs["instrument_manufacturer"] = "Zeiss"
        f.create_dataset("/exchange/data", data=projections)
        f.create_dataset("/exchange/theta", data=angles)
        # ... flats, darks, etc.
    return volume, projections, angles
```

Then a test:

```python
# tests/test_my_vendor.py
from data_import.importer import DataImporter
from tests.fixtures.generate_my_vendor import generate

def test_detects_my_vendor(tmp_path):
    path = tmp_path / "scan.h5"
    generate(path)
    plugin = DataImporter().detect(path, format="hdf5")
    assert plugin.name == "my_vendor_hdf5"
```

**Don't forget**: if you add a new vendor module, also add it to the
import + reload list in `tests/conftest.py`'s `_fresh_registry` fixture
(both the `from data_import.vendors import ...` line and the matching
`importlib.reload(...)` call). This project hit a real bug once where a
new/updated plugin module wasn't included there, and silently stopped
being registered after the first test ran — see the comment in
`conftest.py` for the full explanation of why `importlib.reload()`
needs the submodule directly, not the parent package.

## For TIFF vendors

The same idea applies with `TiffSignature` instead of `Hdf5Signature` —
see `data_import/vendors/synthetic.py`'s `SyntheticTiffPlugin` for a
complete worked example (filename patterns, sidecar JSON requirements,
reading a metadata sidecar file in `parse_metadata()`).

## Reference examples already in the codebase

| File | What it demonstrates |
|---|---|
| `vendors/synthetic.py` | Simplest possible case, both hdf5 and tiff, fixed known paths |
| `vendors/hypothetical_synchrotron.py` | A vendor that includes flats and voxel size, so `voxel_size` is read from the file rather than always declared missing |
| `vendors/generic.py` | Low-specificity fallback plugins — good reference for how specificity/fallback behavior works |
