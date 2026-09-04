# The realistic phantom generator

`tests/fixtures/realistic_phantom/` is a standalone CT data simulator:
a Shepp-Logan phantom projected through a simulated imperfect detector,
with center-of-rotation deviation, tilt, flat/dark-field defects, and
noise all controllable and all with a known ground truth. It exists to
test the rest of the pipeline — normalization, ring removal, and
(later) center-of-rotation finding, tilt correction, and reconstruction
— against data that actually has the problems those stages are meant
to fix, not just a geometrically perfect toy volume.

It's independent of the rest of the project: only `numpy`/`scipy` for
the simulation, `h5py` only inside `write_hdf5()` (imported lazily,
same pattern as everywhere else in this codebase). The one place it
reaches into the main package is `to_project_object()`, which is an
explicit convenience, not a hard dependency of the simulation itself.

## The phantom

The object is the (modified) Shepp-Logan phantom — the standard
reference test object for tomography reconstruction, a sum of
overlapping ellipsoids with published parameters. The 2D ellipse table
in `phantom.py` was verified against `skimage.data.shepp_logan_phantom()`
before use (correlation ≈ 0.9999998 at 400×400), so both the parameter
table and the axis-orientation convention are confirmed correct. The 3D
extension (each ellipse becomes an ellipsoid) is a reasonable,
clearly-documented construction, not a claim of matching some specific
published 3D reference — see `phantom.py`'s module docstring for why.

## The forward model

Deliberately the exact inverse of `preprocessing.normalize()`:

```
transmission = exp(-line_integral * attenuation_scale)      # Beer-Lambert
raw          = dark_offset + gain * flat_intensity * transmission + noise
flat_frame   = dark_offset + gain * flat_intensity + noise
dark_frame   = dark_offset + noise
```

`gain` (spatial non-uniformity, scratches, dead/hot pixels) and
`dark_offset` are the *true*, fixed detector patterns, shared by every
projection, flat, and dark frame — exactly like a real detector's
defects are a property of the hardware, not of one exposure. This means
running `preprocessing.normalize()` on this generator's noiseless output
recovers the ideal transmission almost exactly (`tests/test_realistic_phantom.py`
checks this directly, to `rtol=1e-3`) — a real quantitative check on the
pipeline, not just a plausible-looking file.

Ring artifacts aren't simulated as a separate step — they fall directly
out of the fixed-pattern `gain` defects above being present, and
imperfectly corrected, at the same detector position in every angle.
That's the actual physical cause, not a shortcut.

## What's controllable

| Parameter | Controls |
|---|---|
| `shape` | `(height, width, width)` of the volume/detector |
| `n_angles`, `n_flats`, `n_darks` | how many of each frame to generate |
| `cor_deviation` | center-of-rotation offset, pixels (sub-pixel, applied as a shift) |
| `tilt_deg` | detector/rotation-axis tilt, degrees (applied as an in-plane rotation) |
| `attenuation_scale` | converts the phantom's unitless density into optical depth |
| `flat_intensity` | nominal flat-field brightness |
| `dark_level` | nominal dark-field offset |
| `inhomogeneity_strength` | smooth, low-frequency detector gain variation |
| `n_scratches`, `scratch_strength` | thin fixed-column gain dips in the flat field |
| `n_defective_pixels` | dead/stuck-high pixels (half and half) |
| `noise_level` | relative Gaussian noise per frame; `0.0` (default) is fully deterministic |
| `voxel_size`, `seed` | as named |

## Usage

```python
from tests.fixtures.realistic_phantom import generate, write_hdf5, to_project_object

phantom = generate(
    shape=(64, 96, 96), n_angles=200,
    cor_deviation=2.5, tilt_deg=1.0,
    n_scratches=3, n_defective_pixels=5,
)
```

`generate()` does no file I/O — it returns a `PhantomProject`, a plain
dataclass holding the ground truth (`volume`, `true_center_of_rotation`,
`true_tilt_angle`) alongside the simulated data
(`projections`/`flat_field`/`dark_field`/`angles`). From there, two
separate paths depending on what you're testing:

```python
# Fast path: straight into a ProjectObject, no file involved --
# for testing completion/preprocessing without I/O overhead.
project = to_project_object(phantom)

# Full path: write a real file and import it through the actual
# plugin-detection pipeline -- for testing the whole thing end to end.
write_hdf5(phantom, "phantom.h5")
from data_import.importer import DataImporter
project = DataImporter().import_project("phantom.h5", format="hdf5", storage_mode="ram")
```

The written file uses the layout
[`data_import/vendors/generic_parallel_synchrotron.py`](../data_import/vendors/generic_parallel_synchrotron.py)'s
plugin expects — a real vendor plugin, registered and detected exactly
like any other, not a test-only shortcut. It's deliberately a separate
plugin from `hypothetical_synchrotron_hdf5` rather than an extension of
it, since that one never modeled a dark field at all; see the plugin's
own module docstring for the full reasoning.

Checking a later result against ground truth is direct — keep the
original `phantom` around:

```python
found_cor = ...  # once calibration exists
assert abs(found_cor - phantom.true_center_of_rotation) < 0.5
```

## Where to go next

- [`WRITING_A_PLUGIN.md`](./WRITING_A_PLUGIN.md) — how
  `generic_parallel_synchrotron_hdf5` fits the general plugin pattern.
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — how the rest of the codebase
  is structured.
