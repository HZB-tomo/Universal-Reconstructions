# Reconstruction

`reconstruction/` turns normalized sinograms into reconstructed images:
one slice, several, or a whole volume, through a chosen backend, and
writes the result to disk. One backend contract, nine built-in
backends, all parallel-beam only -- see "Built-in backends" below.

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

Every returned value is an attenuation coefficient in `float32`: `1/cm`
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

## Built-in backends

All nine wrap `algotom`, and all are parallel-beam only. Every one
past `algotom_fbp` was added by wrapping a function algotom already
exposes -- none of this is a new reconstruction implementation.

| Backend | Wraps | Device | Extra dependency |
|---|---|---|---|
| `algotom_fbp` | `fbp_reconstruction` | CPU/GPU | `algotom` (via the `preprocessing` extra) |
| `algotom_gridrec` | `gridrec_reconstruction` | CPU only | `algotom` **and** `tomopy` (conda-only, no pip wheel -- see below) |
| `algotom_bpf` | `bpf_reconstruction` | CPU/GPU | `algotom` |
| `algotom_dfi` | `dfi_reconstruction` | CPU only | `algotom` |
| `algotom_bp` | `astra_reconstruction(method="BP"/"BP_CUDA")` | CPU/GPU | `algotom` **and** `astra-toolbox` (the `reconstruction_extra` extra) |
| `algotom_sirt` | same, `"SIRT"` | CPU/GPU | same |
| `algotom_sart` | same, `"SART"` | CPU/GPU | same |
| `algotom_art` | same, `"ART"` | CPU/GPU | same -- see the note below, `num_iter` behaves very differently here |
| `algotom_cgls` | same, `"CGLS"` | CPU/GPU | same |

FBP/gridrec/BPF/DFI are analytic (one pass, fast); BP/SIRT/SART/ART/CGLS
are iterative except BP itself (slower, but handle noisy/sparse/
limited-angle data better) -- iteration count is `num_iter` in
`**options`, algotom's own default of `1` is almost always too few.

`algotom_art` is a trap if used like the other three: ASTRA's plain
`'ART'` counts individual ray (row) updates, not full sweeps over every
projection the way SIRT/SART/CGLS's `num_iter` does. Confirmed directly
(`tests/test_reconstruction.py`): the same `num_iter=50` that works
fine for SIRT/SART/CGLS reconstructs ART to exactly zero on one test
case and barely-started on another. Pass something on the order of
`n_projections * width` (one full sweep-equivalent) instead. There is
no warning from ASTRA when this is too low.

`algotom_gridrec` needs `tomopy`, which `gridrec_reconstruction()`
defers to entirely (confirmed directly: it's a thin wrapper, not an
algotom-native implementation) -- and TomoPy has no pip wheel at all,
conda-only (`conda install -c conda-forge tomopy`). For that reason
it's deliberately left out of the GUI's reconstruction-algorithm list
(see `gui/adjustment_panel.py`); it's still a registered backend,
usable directly via the CLI/scripting API if you've set tomopy up
yourself.

`resolve_device(backend, device)` reports which device a call will
actually use (or did use), accounting for real hardware, not just the
`device` hint -- see `ReconstructionBackend.actual_device()`. Every
backend but `algotom_gridrec`/`algotom_dfi` (no GPU path in algotom at
all) checks the same `numba.cuda.is_available()` that `fbp_reconstruction()`/
`astra_reconstruction()` use internally, so it can't disagree with what
actually happened.

## Units

- If `geometry.detector_pixel_size` is set (stored in micrometers),
  the result is in inverse centimeters (1/cm) -- the conventional unit
  for a linear attenuation coefficient in x-ray physics, not the
  micrometers `detector_pixel_size` itself is stored in.
- If `detector_pixel_size` is `None`, the result is in "per pixel"
  units (1/px), not comparable across datasets with different pixel
  sizes.

A backend's underlying algorithm produces `mu * pixel_size`, in "per
pixel" units, regardless of library. `reconstruct()` divides by
`detector_pixel_size` (giving 1/um) then multiplies by `api._UM_PER_CM`
(1e4, since 1 cm = 1e4 um) to give `mu` in 1/cm, once, centrally. A
backend never needs to know about `detector_pixel_size` or this
conversion.

This unit is only physically meaningful when `detector_pixel_size`
is a real, measured value -- not `core/known_fields.py`'s silent
`voxel_size` default of `1.0` um for a vendor plugin that never
reported a real pixel size (`detector_pixel_size` derives from
`voxel_size` for parallel-beam; see `completion/derivers.py`). Check
`DataInterface.get_metadata_source("voxel_size") == "file"` before
trusting the 1/cm scale of a given project's result; the GUI's Basic
settings panel does this already (see doc/GUI_ARCHITECTURE.md) and
flags it in the reconstruct status message otherwise.

`algotom.rec.reconstruction.fbp_reconstruction`'s signature has no
pixel-size parameter (`ratio` is a circular field-of-view mask), so its
raw output is "per pixel."

The returned array is always `float32`, enforced centrally in
`reconstruct()` regardless of what a backend returns internally.
`fbp_reconstruction` itself always returns `float32`, even given a
`float64` sinogram as input.

## `to_uint16()`

```python
from reconstruction import to_uint16

converted, clip_record = to_uint16(images, clip_min=0.0, clip_max=0.05)
# clip_record == {"clip_min": 0.0, "clip_max": 0.05, "source_dtype": "float32"}
```

Clips a volume to `[clip_min, clip_max]`, then linearly rescales that
range to the full `uint16` range `[0, 65535]`. With no `clip_min`/
`clip_max`, defaults to the volume's own min/max. Returns the converted
array alongside a record of the exact bounds used, since those are
needed to recover approximate physical values later; put it straight
into `save_volume()`'s `settings` (e.g. `settings["dtype_clip"] =
clip_record`).

## `save_volume()`

```python
from reconstruction import save_volume

save_volume(images, "out/volume.h5", format="hdf5")
save_volume(images, "out/volume_tiff", format="tiff")

save_volume(
    images, "out/volume.h5", format="hdf5",
    settings={
        "backend": "algotom_fbp", "center": correct_center,
        "tilt_deg": tilt_result.tilt_deg, "ring_correction": True,
        "dtype_clip": clip_record,
    },
)
```

Writes a reconstructed volume (or a single 2D image) to disk.
`format="tiff"` writes one file per slice into the folder at `path`
(created if needed), named `slice_00000.tif`, `slice_00001.tif`, ...;
this is the same folder-of-files layout `data_import`'s `TiffBackend`
reads, so the output round-trips back through `DataImporter`.
`format="hdf5"` writes a single dataset, at `dataset_path` (default
`"/volume"`), inside the HDF5 file at `path`.

`settings` is an open dict recording anything about how this volume was
produced: reconstruction backend and options, center, tilt, whether
ring correction ran, crop bounds, `to_uint16()`'s clip record, or
acquisition metadata pulled from a `ProjectObject` via its
`export_metadata()` method (`core/project.py`) -- merge that dict with
any manual entries yourself before passing it in, e.g.
`settings={**project.export_metadata(), "backend": "algotom_fbp"}`.
Written as JSON: an HDF5 attribute (`reconstruction_settings`, on the
dataset) for `format="hdf5"`, or a `metadata.json` file alongside the
slices for `format="tiff"`. Values must be JSON-serializable; numpy
scalars and arrays are converted automatically. Omitted entirely if
`settings` is `None` or empty.

Neither format records units on its own; carry that through `settings`
too if this volume will outlive the session that made it. See "Units"
above for what a volume's raw values mean.

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
