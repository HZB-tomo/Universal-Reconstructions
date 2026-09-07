# Reconstruction

`reconstruction/` turns normalized sinograms into reconstructed images:
one slice, several, or a whole volume, through a chosen backend. One
public function, one backend contract, one built-in backend
(`algotom_fbp`, parallel-beam only).

`reconstruct()` takes plain arrays, not a `ProjectObject`. The caller
prepares the sinogram, angles, center, and geometry beforehand.

## `reconstruct()`

```python
from reconstruction import Geometry, reconstruct

image = reconstruct(sinogram, angles, center, geometry, backend="algotom_fbp")
```

The returned shape matches the input shape:

| You pass | You get back |
|---|---|
| a single `(n_angles, width)` sinogram | a single `(height, width)` image |
| a stack, `(n_angles, n_slices, width)` | a matching stack, `(n_slices, height, width)` |

Every returned value is an attenuation coefficient in `float32`: `1/um`
if `geometry.detector_pixel_size` is set, `1/px` otherwise. See "Units"
below.

The input stack is angle-first, `(n_angles, n_slices, width)`, the same
shape `preprocessing.normalize_rows()` returns:

```python
normalized = normalize_rows(project, some_rows, reference)  # (n_angles, len(some_rows), width)
images = reconstruct(normalized, angles, center, geometry, backend="algotom_fbp")  # (len(some_rows), height, width)
```

The output stack is slice-first, `(n_slices, height, width)`, matching
this codebase's volume convention (e.g.
`tests/fixtures/realistic_phantom/phantom.py`'s `(height, width, width)`
phantom).

### Parameters

- **`sinogram`** -- a single sinogram or a stack. Normalized
  (transmission) data, not raw detector counts and not already
  `-log()`'d; a backend applies that internally if needed.
- **`angles`** -- degrees.
- **`center`** -- a scalar, shared by every sinogram in the stack, or
  an array-like with one value per sinogram.
- **`geometry`** -- a `Geometry(geometry_type=..., source_to_detector_distance=...,
  source_to_object_distance=..., detector_pixel_size=...)`. Field names
  match `core/known_fields.py`. `reconstruct()` raises `ValueError` if
  the chosen backend does not support the requested `geometry_type`.
  `detector_pixel_size` also determines the returned array's unit; see
  "Units" below.
- **`backend`** -- a registered backend name or a
  `ReconstructionBackend` instance.
- **`device`** -- `"cpu"`, `"gpu"`, or `None`. A hint; a backend may
  honor it, ignore it, or use its own default when `None`.
- **`**options`** -- forwarded to the backend, algorithm-specific.

## Units

- If `geometry.detector_pixel_size` is set (stored in micrometers),
  the result is in inverse micrometers (1/um).
- If `detector_pixel_size` is `None`, the result is in "per pixel"
  units (1/px), not comparable across datasets with different pixel
  sizes.

A backend's underlying algorithm produces `mu * pixel_size`, in "per
pixel" units, regardless of library. `reconstruct()` divides by
`detector_pixel_size` once, centrally, to produce `mu` in 1/um. A
backend never needs to know about `detector_pixel_size`.

`algotom.rec.reconstruction.fbp_reconstruction`'s signature has no
pixel-size parameter (`ratio` is a circular field-of-view mask), so its
raw output is "per pixel."

The returned array is always `float32`, enforced centrally in
`reconstruct()` regardless of what a backend returns internally.
`fbp_reconstruction` itself always returns `float32`, even given a
`float64` sinogram as input.

## Writing a backend

```python
from reconstruction.backend import ReconstructionBackend, register_backend

@register_backend
class MyBackend(ReconstructionBackend):
    name = "my_backend"
    supported_geometries = {"parallel"}  # or {"parallel", "cone"}

    def reconstruct(self, sinograms, angles, center, geometry, device=None, **options):
        ...  # sinograms: (n_angles, n_slices, width)
        return images  # (n_slices, height, width)
```

The contract specifies only shapes in and out; how a backend processes
the batch (looping per slice, a library's native batch API, CPU or
GPU) is internal to the backend.

Add an import line to `reconstruction/backends/__init__.py` to
register a new module.

## `algotom_fbp` backend

Parallel-beam only (`supported_geometries = {"parallel"}`). Wraps
`algotom.rec.reconstruction.fbp_reconstruction`, which takes one
sinogram at a time; this backend loops over the requested stack.

Requires the optional `algotom` package (`pip install .[preprocessing]`).
`get_backend("algotom_fbp")` resolves without it installed; calling
`.reconstruct()` requires the real library.

Converts `angles` from degrees to radians internally, and applies
`-log()` internally (`apply_log=True`, algotom's default).

Reconstructing the Shepp-Logan phantom
(`tests/fixtures/realistic_phantom/`) at moderate resolution through
this backend tops out around correlation 0.7-0.9 against ground truth,
even with a correct center. This matches `algotom`'s own native
round-trip (`algotom.util.simulation.make_sinogram` -> `fbp_reconstruction`)
at the same resolution, consistent with FBP/Gibbs-ringing behavior for
a phantom with sharp discontinuities.

## See also

- [`doc/demo/04_reconstruction.ipynb`](./demo/04_reconstruction.ipynb):
  generates a phantom with a known center-of-rotation deviation,
  reconstructs one slice with the wrong center versus the correct one,
  and reconstructs several slices at once.
- [`REALISTIC_PHANTOM.md`](./REALISTIC_PHANTOM.md): the test-data
  generator this module is tested against.
- [`ARCHITECTURE.md`](./ARCHITECTURE.md): how this fits with the rest
  of the codebase.
