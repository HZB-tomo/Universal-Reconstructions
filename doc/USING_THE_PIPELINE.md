# Using the pipeline

For loading and preparing a dataset in a notebook or script. For
writing a vendor plugin, see [`WRITING_A_PLUGIN.md`](./WRITING_A_PLUGIN.md).
For how the codebase is structured, see [`ARCHITECTURE.md`](./ARCHITECTURE.md).

The pipeline has five stages:

```
import  ->  completion  ->  preprocessing  ->  calibration  ->  reconstruction
```

## 1. Importing a project

```python
from data_import.importer import DataImporter

importer = DataImporter()
project = importer.import_project(
    "my_scan.h5",
    format="hdf5",          # or "tiff", for a folder of tiff files
    storage_mode="ram",     # or "disk"
)
print(project)
```

`format` is `"hdf5"` or `"tiff"`.

`storage_mode` controls whether projection/flat/dark arrays are read
fully into memory (`"ram"`) or left on disk and read lazily
(`"disk"`). Both formats support both modes: `"disk"` never reads
pixel data until something asks for it, so it's cheap to use even just
to check a file's metadata.

### If the file isn't recognized

- `NoVendorMatchError`: no plugin recognized the file. The message
  lists, for every known vendor, which check failed.
- `AmbiguousVendorMatchError`: more than one vendor convention matched
  equally well. Resolve by passing a specific plugin instance via
  `import_project(..., plugin=<instance>)`.

### Reading the actual data

```python
from core.data_interface import DataInterface

di = DataInterface(project)

di.get_projections()          # the full projection stack
di.get_projections(index=0)   # just one angle
di.get_flats()                # None if this project has no flats
di.get_metadata("angles")
```

`di.materialize_to_ram()` converts every disk-backed handle to a RAM
handle.

## 2. Completion — filling in what the file didn't tell you

### Checking what's there

```python
from core.known_fields import status

for row in status(project.metadata):
    print(row.key, row.tier, row.resolved, row.value)
```

Reports on every field the software depends on. See
[`PROJECT_METADATA.md`](./PROJECT_METADATA.md) for field meaning, unit,
and valid values.

### Filling in what can be filled in automatically

```python
import completion

completion.run(project)
```

Applies defaults and derives values from data already present (e.g.
`angles` from `angle_span` and the projection count, or vice versa).
Safe to call repeatedly; resolved fields are left alone. Fields with no
safe default and nothing to derive from (e.g. `geometry_type`) remain
missing.

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

Each call records the answer and re-runs the automatic pass, so one
answer can unlock others. For example, `geometry_type="cone"` makes
cone-beam-only fields like `source_to_detector_distance` newly
required.

```python
completion.run(project)
while project.needs_user_input():
    for key in project.missing_soft_fields():
        value = ...  # input(), a config file, a form
        completion.provide(project, key, value)
```

### What completion does not do

`angles`, `angle_span`, and `detector_pixel_size` have derive logic
(`completion/derivers.py`), but a deriver only helps if its own inputs
are available. If a vendor provides neither `angles` nor `angle_span`,
both stay missing until supplied via `completion.provide()`.
`center_of_rotation` and `tilt_angle` are `computed`-tier and are not
touched by completion; they are the job of the calibration stage.

## 3. Checking completion readiness

```python
project.needs_user_input()   # False once completion is done
```

`status(project.metadata)` gives the full picture.

## 4. Preprocessing — normalization and artifact removal

### Flat/dark correction

```python
from preprocessing import build_reference, normalize, normalize_slice, normalize_rows

reference = build_reference(project)              # reduces flats/darks once
normalize(raw_array, reference)                    # the raw formula, given an array
normalize_slice(project, row=64, reference=reference)          # one sinogram
normalize_rows(project, slice(100, 200), reference=reference)  # a batch of rows
normalize_rows(project, slice(None), reference=reference)      # the entire volume
```

`build_reference()` raises `MissingFlatFieldError` if the project has
no flat field data. A missing dark field is handled two ways:

```python
build_reference(project)                    # no dark data, no known offset -- dark = 0.0
build_reference(project, dark_offset=100.0)  # no dark data, a known constant offset
```

`dark_offset`, when given, takes precedence over any real dark frames
the project has. `reference.dark` is then a scalar instead of a 2D
image.

`-log()` is not applied here; it happens at or inside reconstruction.

### Zinger and ring-artifact removal (optional)

```python
from preprocessing import remove_zinger, remove_ring_artifacts, process_volume

clean = remove_zinger(sinogram, threshold=0.08)
clean = remove_ring_artifacts(sinogram, method="sorting")   # or "filtering"/"fitting"/"normalization"/"all"

# batched across every row of a volume, in parallel:
clean_volume = process_volume(volume, remove_ring_artifacts, method="sorting")
```

Run on `normalize()`'s output. Requires the optional `algotom` package
(`pip install .[preprocessing]` or `pip install algotom`); calling
these without it installed raises `ImportError`.

### Where does the result go?

Nowhere. Every function in `preprocessing/` is pure: array in, array
out. Nothing is written back onto `project`.

## 5. Calibration — finding the center of rotation and tilt

```python
from calibration import find_center, find_tilt, centers_for_rows

middle_row = normalized.shape[1] // 2
center = find_center(normalized[:, middle_row, :], backend="algotom")   # or a manual number
result = find_tilt(normalized, backend="algotom", num_rows=12, margin=4)
result.tilt_deg   # degrees
```

`normalized` is the `(n_angles, n_rows, width)` array
`normalize_rows()` produces.

`find_center()` and `find_tilt()` take plain arrays, not a
`ProjectObject`, and `backend` is always explicit. Neither reconstructs
anything internally; calling either twice on the same input gives the
same answer. See [`CALIBRATION.md`](./CALIBRATION.md) for the method
and for why "tilt" here means rotation-axis tilt, not an in-plane
rotation of the detector.

```python
from calibration import centers_for_rows

rows = list(range(normalized.shape[1]))
centers = centers_for_rows(result.tilt_deg, center, reference_row=middle_row, rows=rows)
```

`reconstruct()` accepts one center per slice, so this array is all
reconstruction needs to account for the tilt.

## 6. Reconstruction

```python
from reconstruction import Geometry, reconstruct

geometry = Geometry(geometry_type=di.get_metadata("geometry_type"))
image = reconstruct(sinogram, project.metadata.get("angles"), center, geometry, backend="algotom_fbp")
```

`reconstruct()` takes plain arrays, not a `ProjectObject`. See
[`RECONSTRUCTION.md`](./RECONSTRUCTION.md).

```python
one_slice = normalize_slice(project, row=64, reference=reference)
image = reconstruct(one_slice, angles, center, geometry, backend="algotom_fbp")           # (height, width)

several = normalize_rows(project, slice(60, 70), reference=reference)
images = reconstruct(several, angles, center, geometry, backend="algotom_fbp")            # (10, height, width)
```

`normalize_rows()`'s output shape, `(n_angles, n_slices, width)`, is
the shape `reconstruct()` expects for a stack.

A wrong center produces a visibly off-axis, asymmetrically cropped
image; see [`doc/demo/04_reconstruction.ipynb`](./demo/04_reconstruction.ipynb)
for a side-by-side. A manual visual check (reconstruct a range of
candidate centers and look at them) is a direct call to `reconstruct()`
with an array of trial centers.

`algotom_fbp` needs the same optional `algotom` package as ring/zinger
removal.

If `geometry.detector_pixel_size` is set (stored in micrometers), the
returned values are attenuation coefficients in inverse micrometers
(1/um). If it's `None`, they are in "per pixel" units. The result is
always `float32`. See [`RECONSTRUCTION.md`](./RECONSTRUCTION.md)'s
"Units" section.

## Full example

```python
from data_import.importer import DataImporter
from core.known_fields import status
import completion
from preprocessing import build_reference, normalize_rows
from calibration import find_center, find_tilt, centers_for_rows
from reconstruction import Geometry, reconstruct

importer = DataImporter()
project = importer.import_project("my_scan.h5", format="hdf5", storage_mode="ram")

completion.run(project)
while project.needs_user_input():
    for key in project.missing_soft_fields():
        value = input(f"enter a value for {key}: ")
        completion.provide(project, key, value)

assert not project.needs_user_input()
for row in status(project.metadata):
    print(f"  {row.key}: {row.value!r}")

reference = build_reference(project)
normalized = normalize_rows(project, slice(None), reference)

geometry = Geometry(geometry_type=project.metadata.get("geometry_type"))
middle_row = normalized.shape[1] // 2
center = find_center(normalized[:, middle_row, :], backend="algotom")
tilt = find_tilt(normalized, backend="algotom", num_rows=12, margin=4)
centers = centers_for_rows(tilt.tilt_deg, center, reference_row=middle_row, rows=range(normalized.shape[1]))
volume = reconstruct(normalized, project.metadata.get("angles"), centers, geometry, backend="algotom_fbp")
```

## See also

- [`doc/demo/02_completion.ipynb`](./demo/02_completion.ipynb)
- [`doc/demo/03_realistic_phantom.ipynb`](./demo/03_realistic_phantom.ipynb)
- [`doc/demo/04_reconstruction.ipynb`](./demo/04_reconstruction.ipynb)
- [`CALIBRATION.md`](./CALIBRATION.md)
- [`RECONSTRUCTION.md`](./RECONSTRUCTION.md)
- [`PROJECT_METADATA.md`](./PROJECT_METADATA.md)
- [`ARCHITECTURE.md`](./ARCHITECTURE.md)
