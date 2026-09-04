# Reconstruction

`reconstruction/` turns normalized sinograms into reconstructed images
-- one slice, several, or a whole volume, through whichever backend
(algorithm/package) you choose. This is a reference for the module as
it exists today: one public function, one backend contract, one
built-in backend (`algotom_fbp`, parallel-beam only).

## Why it doesn't take a `ProjectObject`

Every other stage in this pipeline (`completion`, `preprocessing`)
reads from a `ProjectObject`. `reconstruction.reconstruct()`
deliberately doesn't -- it takes plain arrays, prepared by the caller
beforehand. This was a real design correction, not the original plan:
see the reasoning in `reconstruction/api.py`'s module docstring. Short
version: a center-of-rotation search reconstructs many *candidate*
variants of the same sinogram (each pre-shifted by a different trial
offset -- see `oldscripts/BAMline_recon_v0p2.py`'s `find_CoR()`), not
"row R of some project." That data has no natural `ProjectObject`
expression -- it's a purpose-built array the search loop constructs
for its own purposes. A `ProjectObject`-aware `reconstruct()` couldn't
serve that; a plain array-in-array-out one does, for free, with no
"candidate" concept anywhere in this module.

## `reconstruct()`

```python
from reconstruction import Geometry, reconstruct

image = reconstruct(sinogram, angles, center, geometry, backend="algotom_fbp")
```

One function, not two -- which shape you get back matches which shape
you pass in, numpy-indexing style:

| You pass | You get back |
|---|---|
| a single `(n_angles, width)` sinogram | a single `(height, width)` image |
| a stack, `(n_angles, n_slices, width)` | a matching stack, `(n_slices, height, width)` |

Every returned value is an attenuation coefficient in `float32` --
`1/um` if `geometry.detector_pixel_size` is set, `1/px` otherwise. See
"Units" below.

**The stack's axis order is deliberate and worth being precise about**:
the *input* stack is angle-first, `(n_angles, n_slices, width)` --
exactly what `preprocessing.normalize_rows()` already returns, so its
output can be hand straight to `reconstruct()` with no reshaping:

```python
normalized = normalize_rows(project, some_rows, reference)  # (n_angles, len(some_rows), width)
images = reconstruct(normalized, angles, center, geometry, backend="algotom_fbp")  # (len(some_rows), height, width)
```

The *output* stack is slice-first, `(n_slices, height, width)` --
matching this codebase's own volume convention instead (e.g.
`tests/fixtures/realistic_phantom/phantom.py`'s `(height, width, width)`
phantom). Input and output are different kinds of stacks (sinograms
vs. reconstructed images), so matching two different existing
conventions here is intentional, not an inconsistency.

### Parameters

- **`sinogram`** -- a single sinogram or a stack, as above. Always
  normalized (transmission) data, never raw detector counts and never
  already `-log()`'d -- every backend applies that internally if it
  needs to (see `algotom_fbp` below).
- **`angles`** -- degrees, matching this project's convention
  everywhere else (`core/known_fields.py`'s `angles` field). Backends
  convert to whatever unit their underlying library expects internally.
- **`center`** -- a scalar (one center shared by every sinogram in the
  stack) or an array-like with one value per sinogram. The array form
  is what a per-row calibration search would use if it supplies
  already-different, already-known centers per physical row, as
  opposed to the pre-shift-and-share-one-center trick described above.
- **`geometry`** -- a `Geometry(geometry_type=..., source_to_detector_distance=...,
  source_to_object_distance=..., detector_pixel_size=...)`. Field names
  match `core/known_fields.py` exactly; a caller working from a
  `ProjectObject` builds one from `project.metadata`. `reconstruct()`
  raises `ValueError` immediately if the chosen backend doesn't support
  the requested `geometry_type`, before calling it. `detector_pixel_size`
  also determines the returned array's unit -- see "Units" below.
- **`backend`** -- a name (looked up via `get_backend()`) or a
  `ReconstructionBackend` instance directly. Always explicit -- nothing
  here auto-detects a backend the way vendor import plugins auto-detect
  a file's vendor; there's nothing in the data to detect it from.
- **`device`** -- `"cpu"`, `"gpu"`, or `None`. A *hint*, not a
  guarantee: a backend is free to honor it, ignore it, or fall back to
  its own default when `None`. This module makes no promise of
  automatic cost-based CPU/GPU selection -- a backend that wants that
  has to implement it itself.
- **`**options`** -- forwarded straight to the backend, algorithm-specific
  (filter name, iteration count, whatever that backend's library needs).

## Units

The returned array is a real attenuation coefficient, in a unit chosen
automatically -- there's no parameter for this, and deliberately no
choice to offer:

- **If `geometry.detector_pixel_size` is given** (this project always
  stores it in micrometers -- see `Geometry`), the result is in
  **inverse micrometers (1/um)**.
- **If `detector_pixel_size` is `None`**, the result is in **"per
  pixel" units (1/px)** -- not physically comparable across datasets
  with different pixel sizes, since there's nothing to convert by.

Why this is automatic rather than a setting: every backend's
underlying algorithm (FBP or otherwise) inverts the line-integral
equation `-log(T) = integral of mu dl` assuming unit pixel spacing --
that's just what the math naturally produces, regardless of library.
The result is `mu * pixel_size`, not `mu` itself. Dividing by the real
`detector_pixel_size` is the one and only correct way to recover `mu`
in physical units; there's no second reasonable convention to pick
between, so `reconstruct()` (in `reconstruction/api.py`) does this
conversion itself, once, centrally, for every backend -- no backend
needs to know about `detector_pixel_size` at all. A backend's
`reconstruct()` always returns "per pixel" values; converting to 1/um
happens strictly afterward, at the public API boundary.

This was verified directly against `algotom.rec.reconstruction.fbp_reconstruction`:
its signature has no pixel-size or physical-scale parameter (`ratio`
is only a circular field-of-view mask), confirming its raw output is
"per pixel," exactly as assumed above.

**dtype**: the returned array is always `float32`, regardless of what
a backend returns internally -- enforced centrally in
`reconstruct()`, the same place unit conversion happens. This isn't
an arbitrary choice: most GPUs (consumer and datacenter alike) have
dramatically lower float64 throughput than float32 -- often 1/32 to
1/64 on consumer hardware -- and float32's ~7 significant decimal
digits already far exceed the precision of real detector data or any
attenuation coefficient derived from it. This matches
`fbp_reconstruction` itself, empirically confirmed to always return
`float32` even when handed a `float64` sinogram as input.

## Writing a backend

Exactly one method to implement:

```python
from reconstruction.backend import ReconstructionBackend, register_backend

@register_backend
class MyBackend(ReconstructionBackend):
    name = "my_backend"
    supported_geometries = {"parallel"}  # or {"parallel", "cone"}

    def reconstruct(self, sinograms, angles, center, geometry, device=None, **options):
        ...  # sinograms: (n_angles, n_slices, width) -- see above
        return images  # (n_slices, height, width)
```

**How you handle the batch is entirely up to you** -- that's the point
of the contract only specifying shapes in and out, nothing about
mechanism. A CPU-bound library benefits from parallelizing across
independent slices (a process pool over the slice axis, the way
`algotom.util.parallel_process_slices` already works, reused in
`preprocessing/corrections.py`). A GPU-batch-native library (or a
future one -- possibly not even Python) should hand its whole batch to
one internal call instead: launching one small GPU call per slice pays
real, fixed overhead every time (host/device transfer, kernel launch
latency) and rarely saturates the GPU's own parallelism, since there's
normally one physical device and external parallelism over it just
queues rather than actually parallelizing. Neither strategy requires
changing this contract -- only the body of `reconstruct()`.

Register the new module the same way vendor plugins do: add an import
line to `reconstruction/backends/__init__.py`.

## `algotom_fbp` -- the built-in backend

Parallel-beam only (`supported_geometries = {"parallel"}`). Wraps
`algotom.rec.reconstruction.fbp_reconstruction`, which only takes one
sinogram at a time -- this backend loops over the requested stack, one
call per slice. That loop is **not parallelized** (no process pool, no
GPU batching) -- it's the simplest correct reference implementation,
not a claim of being the fastest possible one. Real batch/parallel
performance work belongs in a backend's `reconstruct()` body (this one
or a future one), not in the contract -- see the section above.

Needs the optional `algotom` package (`pip install .[preprocessing]`,
the same extra `preprocessing/corrections.py` uses). `get_backend("algotom_fbp")`
resolves fine without it installed -- only actually calling
`.reconstruct()` needs the real library, at which point a plain
`ImportError` tells you what to install.

Converts `angles` from degrees to radians internally, and applies
`-log()` internally by default (`apply_log=True`, algotom's own
default, not overridden by this backend) -- consistent with every real
pipeline checked in `oldscripts/`, where `-log()` always happens at/inside
reconstruction, never as part of flat/dark correction (see
`preprocessing/normalization.py`'s module docstring for the fuller
version of that finding).

### A real, verified reconstruction-quality ceiling, not a bug

Reconstructing the Shepp-Logan phantom (`tests/fixtures/realistic_phantom/`)
at moderate resolution through this backend tops out around
correlation 0.7-0.9 against ground truth, even with a perfectly correct
center -- confirmed directly against `algotom`'s own native round-trip
(`algotom.util.simulation.make_sinogram` → `fbp_reconstruction`) hitting
the same ceiling. That's genuine FBP/Gibbs-ringing behavior for a
phantom with sharp discontinuities, not a defect in this backend or the
phantom generator -- see `tests/fixtures/realistic_phantom/simulate.py`'s
`project_parallel()` docstring for the full verification story
(including a real bug that verification caught and fixed: the original
rotate-and-sum forward projector had the rotation sign backwards and
summed the wrong axis).

## Where to go next

- [`doc/demo/04_reconstruction.ipynb`](./demo/04_reconstruction.ipynb) --
  generates a phantom with a known center-of-rotation deviation,
  reconstructs one slice with the wrong center vs. the correct one, and
  reconstructs several slices at once through the same call.
- [`REALISTIC_PHANTOM.md`](./REALISTIC_PHANTOM.md) -- the test-data
  generator this module is verified against.
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) -- how this fits with the rest
  of the codebase.
