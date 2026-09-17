# Writing a new vendor plugin

Adding support for a new instrument touches two files, plus optionally
a test.

## What you're building

A plugin answers three questions about one vendor's file convention:

1. **Is this my file?** A signature: a checklist of conditions
   (required hdf5 paths/attributes, or tiff filename/sidecar patterns).
2. **Where's everything?** A `PathMap` pointing to projections, flats,
   darks, angles.
3. **What does the metadata mean?** Reading values and sorting them
   into the four tiers (`mandatory_hard`, `mandatory_soft`, `optional`,
   `computed`).

A plugin never opens a file itself and never chooses RAM vs disk; see
`ARCHITECTURE.md`.

## Step 1 — figure out the file layout

Example, for a fictional instrument that writes:

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

Notes:

- Signature specificity is the number of conditions checked. This
  example checks 4 paths plus 1 attribute, specificity 5. It needs to
  beat the generic fallback (specificity 1) and not tie with another
  vendor plugin.
- Tiering depends on the vendor's actual behavior. `voxel_size` is read
  conditionally here because this vendor sometimes omits it; a vendor
  that always includes it should read it unconditionally, as
  `hypothetical_synchrotron.py` does.
- `parse_metadata()` reads only small values (scalars, short arrays
  like the angle list). Projections/flats/darks arrays are read later
  by the backend, using the paths returned from `get_path_map()`.

## Step 3 — register it

Add one import line to `data_import/vendors/__init__.py`:

```python
from data_import.vendors import my_vendor  # noqa: F401
```

`DataImporter` tries this plugin automatically whenever
`import_project(..., format="hdf5")` is called.

## Step 4 (recommended) — write a test-data generator and test

The pattern used elsewhere in this codebase
(`tests/fixtures/generate_synthetic.py`,
`tests/fixtures/generate_hypothetical_synchrotron.py`) is a small
script that builds a synthetic volume, projects it, and writes it in
the new vendor's layout.

```python
# tests/fixtures/generate_my_vendor.py
import h5py
from tests.fixtures.generate_synthetic import make_volume, project_parallel

def generate(path, shape=(32, 24, 24), n_angles=10, seed=None):
    volume = make_volume(shape=shape, seed=seed)
    projections, angles = project_parallel(volume, n_angles=n_angles)
    with h5py.File(path, "w") as f:
        f.attrs["instrument_manufacturer"] = "Zeiss"
        f.create_dataset("/exchange/data", data=projections)
        f.create_dataset("/exchange/theta", data=angles)
        # ... flats, darks, etc.
    return volume, projections, angles
```

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

If a new vendor module is added, also add it to the import and reload
list in `tests/conftest.py`'s `_fresh_registry` fixture: both the
`from data_import.vendors import ...` line and the matching
`importlib.reload(...)` call. Without this, the plugin stops being
registered after the first test runs.

## For TIFF vendors

Use `TiffSignature` instead of `Hdf5Signature`; see
`data_import/vendors/synthetic.py`'s `SyntheticTiffPlugin` for a
complete example (filename patterns, sidecar JSON requirements, reading
a metadata sidecar file in `parse_metadata()`).

## When flats/darks share a dataset with projections

Not every vendor gives projections, flats, and darks their own
separate dataset. `PathMap` has `projections_range`/`flats_range`/
`darks_range` (each `Optional[slice]`, default `None` meaning "the
whole dataset", so every plugin that doesn't need this is unaffected)
for exactly this: point two roles at the *same* path, and give each its
own slice of it. `Hdf5Backend`/`TiffBackend` apply the slice before
wrapping the result in a `DataHandle`, so nothing downstream needs to
know this happened -- `DataInterface.get_projections()` etc. see a
clean, zero-based array either way.

`vendors/bamline.py` is the real example: BAMline's on-the-fly scans
write one dataset covering the whole acquisition (a stationary
flat-field plateau, the real rotation, then another stationary
stretch), not separate flats/projections files. `get_path_map()` works
out where the plateau ends and where a clean ~180 deg window begins
from the angle data itself (no fixed frame count), then returns both
roles pointing at the one path with different ranges. Only a slice
`step` of 1 is supported (`core/handles.py`'s `Hdf5DiskHandle`) -- the
one shape this exists for is a contiguous frame range, not a strided
one.

## When the file carries no vendor-identifying attribute at all

Not every vendor writes something as convenient as `beamline ==
"my_vendor"`. `vendors/bamline.py`'s files have no such marker at all;
its `Hdf5Signature` instead combines a few always-present,
vendor-distinctive paths with a `custom_predicate` that checks a
structural property (at least one of several candidate rotation-stage
attributes actually shows real angular movement, not encoder noise on
a stage that never turned) -- still deterministic, still countable
toward specificity, just not a single fixed path or attribute value.

## Reference examples in the codebase

| File | What it demonstrates |
|---|---|
| `vendors/synthetic.py` | Simplest case, both hdf5 and tiff, fixed known paths |
| `vendors/hypothetical_synchrotron.py` | A vendor with flats and voxel size read from the file |
| `vendors/generic_parallel_synchrotron.py` | A vendor with flats and darks, and a field (`geometry_type`) known unconditionally from the vendor's convention; paired with [`REALISTIC_PHANTOM.md`](./REALISTIC_PHANTOM.md)'s test-data generator |
| `vendors/bamline.py` | Flats and projections sharing one dataset via `PathMap` ranges; picking the right one of several candidate metadata paths by data content, not by name; no vendor-identifying attribute at all, so the signature leans on structure instead; no darks in the file at all (a fixed per-detector `dark_offset` constant instead, see `preprocessing/normalization.py`) |
| `vendors/generic.py` | Low-specificity fallback plugins |
