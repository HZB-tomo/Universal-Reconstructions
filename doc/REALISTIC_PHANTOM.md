# The realistic phantom generator

`tests/fixtures/realistic_phantom/` is a CT data simulator: a
Shepp-Logan phantom projected through a simulated imperfect detector,
with center-of-rotation deviation, rotation-axis tilt, flat/dark-field
defects, and noise, all controllable and all with a known ground
truth.

Depends only on `numpy`/`scipy`, plus `h5py` inside `write_hdf5()` and
`tifffile` inside `write_tiff()`. `to_project_object()` is the one
place it reaches into the main package.

## The phantom

The object is the (modified) Shepp-Logan phantom: a sum of overlapping
ellipsoids with published parameters. The 2D ellipse table in
`phantom.py` matches `skimage.data.shepp_logan_phantom()` (correlation
0.9999983 at 400x400). The 3D extension is not a published table; each
ellipse is extruded into an ellipsoid with z half-axis `min(a, b)`.

## The forward model

The inverse of `preprocessing.normalize()`:

```
transmission = exp(-line_integral * attenuation_scale)      # Beer-Lambert
raw          = dark_offset + gain * flat_intensity * transmission + noise
flat_frame   = dark_offset + gain * flat_intensity + noise
dark_frame   = dark_offset + noise
```

`gain` (spatial non-uniformity, scratches, dead/hot pixels) and
`dark_offset` are the fixed detector patterns, shared by every
projection, flat, and dark frame. `preprocessing.normalize()` applied
to this generator's noiseless output recovers the ideal transmission to
`rtol=1e-3` (`tests/test_realistic_phantom.py`).

Ring artifacts are not simulated as a separate step; they follow from
the fixed-pattern `gain` defects being imperfectly corrected at the
same detector position in every angle.

## What's controllable

| Parameter | Controls |
|---|---|
| `shape` | `(height, width, width)` of the volume/detector |
| `n_angles`, `n_flats`, `n_darks` | how many of each frame to generate |
| `angle_span` | degrees the `n_angles` angles are evenly spaced over, starting at 0. Default `180.0`; less than `180` produces a limited-angle scan |
| `cor_deviation` | center-of-rotation offset at the middle row, pixels |
| `tilt_deg` | rotation-axis tilt (axis lean), degrees; the effective center drifts linearly with row |
| `attenuation_scale` | converts the phantom's unitless density into optical depth; `None` (default) solves for it so the single most-attenuating ray transmits `target_min_transmission` of the free beam, matching how a real acquisition's exposure is set up (see `simulate.py`'s `attenuation_scale_for_min_transmission()`) |
| `target_min_transmission` | the worst-case ray's transmission fraction `attenuation_scale` is solved for; default `0.14`, matching real CT practice (at least ~10-15%) |
| `flat_intensity` | nominal flat-field brightness |
| `dark_level` | nominal dark-field offset |
| `inhomogeneity_strength` | smooth, low-frequency detector gain variation |
| `n_scratches`, `scratch_strength` | thin fixed-column gain dips in the flat field |
| `n_defective_pixels` | dead/stuck-high pixels (half and half) |
| `noise_level` | relative Gaussian noise per frame; `0.0` (default) is deterministic |
| `voxel_size`, `seed` | as named |

## Usage

```python
from tests.fixtures.realistic_phantom import generate, write_hdf5, write_tiff, to_project_object

phantom = generate(
    shape=(64, 96, 96), n_angles=200,
    cor_deviation=2.5, tilt_deg=1.0,
    n_scratches=3, n_defective_pixels=5,
)
```

`generate()` does no file I/O; it returns a `PhantomProject`, a
dataclass holding ground truth (`volume`, `true_center_of_rotation`,
`true_tilt_angle`) alongside the simulated data
(`projections`/`flat_field`/`dark_field`/`angles`).

```python
# Straight into a ProjectObject, no file involved:
project = to_project_object(phantom)

# Write a file and import it through the plugin-detection pipeline:
write_hdf5(phantom, "phantom.h5")
from data_import.importer import DataImporter
project = DataImporter().import_project("phantom.h5", format="hdf5", storage_mode="ram")

# Or as a folder of TIFF files, same phantom:
write_tiff(phantom, "phantom_tiff")
project = DataImporter().import_project("phantom_tiff", format="tiff", storage_mode="ram")
```

Both use the layout
[`data_import/vendors/generic_parallel_synchrotron.py`](../data_import/vendors/generic_parallel_synchrotron.py)'s
two plugins (`generic_parallel_synchrotron_hdf5`, `generic_parallel_synchrotron_tiff`)
expect, registered and detected like any other vendor plugin.

Checking a result against ground truth:

```python
found_cor = ...
assert abs(found_cor - phantom.true_center_of_rotation) < 0.5
```

## See also

- [`WRITING_A_PLUGIN.md`](./WRITING_A_PLUGIN.md)
- [`ARCHITECTURE.md`](./ARCHITECTURE.md)
