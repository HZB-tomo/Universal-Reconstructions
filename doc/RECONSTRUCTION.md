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

Ten backends, all parallel-beam only. Nine wrap `algotom`; `gridrec` is
this project's own implementation (see below). Each backend declares the
packages it needs (`ReconstructionBackend.requires`); one whose packages are
missing stays registered but is left out of the GUI's algorithm list, and
`reconstruction.list_backends()` returns only the usable ones.

| Backend | Wraps | Device | Extra dependency |
|---|---|---|---|
| `algotom_fbp` | `fbp_reconstruction` | CPU/GPU | `algotom` (via the `preprocessing` extra) |
| `gridrec` | `urct_gridrec` (own implementation) | CPU; GPU on request (experimental) | `urct_gridrec` (the `gridrec` extra) |
| `algotom_gridrec` | `gridrec_reconstruction` | CPU only | `algotom` **and** `tomopy` (conda-only, no pip wheel -- see below) |
| `algotom_bpf` | `bpf_reconstruction` | CPU/GPU | `algotom` |
| `algotom_dfi` | `dfi_reconstruction` | CPU only | `algotom` |
| `algotom_bp` | `astra_reconstruction(method="BP"/"BP_CUDA")` | CPU/GPU | `algotom` **and** `astra-toolbox` (the `reconstruction_extra` extra) |
| `algotom_sirt` | same, `"SIRT"` | CPU/GPU | same |
| `algotom_sart` | same, `"SART"` | CPU/GPU | same |
| `algotom_art` | same, `"ART"` | CPU/GPU | same -- see the note below, `num_iter` behaves very differently here |
| `algotom_cgls` | same, `"CGLS"` | CPU/GPU | same |

FBP/gridrec/BPF/DFI are analytic (one pass, fast); of BP/SIRT/SART/
ART/CGLS, SIRT/SART/ART/CGLS are iterative (BP itself isn't -- see
`ReconstructionBackend.is_iterative`) -- iteration count is `num_iter`
in `**options`, algotom's own default of `1` is almost always too few.
In the GUI, `num_iter` has its own "Iterations" control in Basic
settings, shown only for those four backends -- see
doc/GUI_ARCHITECTURE.md's "Basic settings" section; via the CLI/
scripting API, pass `num_iter` explicitly in `reconstruction_options`.

`algotom_art` behaves differently from the other three: ASTRA's plain
`'ART'` counts individual ray (row) updates, not full sweeps over every
projection the way SIRT/SART/CGLS's `num_iter` does. `num_iter=50`,
which is enough for SIRT/SART/CGLS, reconstructs ART to zero or
barely-started (see `tests/test_reconstruction.py`). Pass something on
the order of `n_projections * width` (one full sweep-equivalent)
instead. ASTRA gives no warning when this is too low.

`algotom_gridrec` needs `tomopy` -- `gridrec_reconstruction()` is a
thin wrapper around it, not an algotom-native implementation -- and
TomoPy has no pip wheel, conda-only (`conda install -c conda-forge
tomopy`). It only appears in the GUI's algorithm list when tomopy is
importable; the `gridrec` backend below does the same job without conda.

### `gridrec`

Fourier-gridding filtered backprojection (Dowd et al., J. Synchrotron
Rad. 6, 255, 1999), implemented in the separate `urct_gridrec` package
(`gridrec_project/`; how it works, tests and benchmarks are in its README).
Per slice it costs about width^2 log(width) instead of backprojection's
width^2 x angles, so it is faster the larger the scan:

| Phantom width | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|
| speedup over `algotom_fbp` (CPU) | 14x | 17x | 32x | 56x |

Real project files in `gui_samples/` ran 5-16x faster. Accuracy against the
true image is the same or slightly better than FBP's; output has the same
scale (1/cm after unit conversion). It is reproducible: the same input gives
the same bits, because the gridding is split between threads without any
shared writes.

Options (`reconstruction_options`): `filter_name` (`ramp`, `hann` default,
`hamming`, `cosine`, `shepp-logan`), `oversampling` (default 1.25: padded
length / width), `kernel_width` (default 6), `ratio` (circular field of view,
1.0 default, `None` to keep the corners), `apply_log` (default True: input is
transmission).

Device: `Auto`/`CPU` use the CPU, which is the tested path. `GPU` runs the
FFTs on CuPy (`pip install -e ".[gridrec_gpu]"`) and the gridding as a
`numba.cuda` kernel; that path is checked against the CPU on numba's CUDA
simulator but has not been timed on a real GPU, so it is never chosen
automatically.

`resolve_device(backend, device)` reports which device a call will
actually use (or did use), accounting for real hardware, not just the
`device` hint -- see `ReconstructionBackend.actual_device()`. Every
backend but `algotom_gridrec`/`algotom_dfi` (no GPU path in algotom)
checks the same `numba.cuda.is_available()` that
`fbp_reconstruction()`/`astra_reconstruction()` use internally.
`gridrec` is CPU unless asked for the GPU explicitly.

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

`StreamingVolumeWriter` is `save_volume()`'s incremental counterpart --
see "Streaming" below.

## Streaming

`reconstruct()`/`to_uint16()`/`save_volume()` above all take and
return one full in-memory array -- fine for a project that fits in
RAM, but that's a hard ceiling on how large a project this pipeline
can reconstruct at all. `ProjectConfig.stream=True` switches
`PipelineRun.run_all()` over to `PipelineRun._run_streaming()`
(`orchestration/pipeline.py`) instead of the normal
`preprocess()`→`calibrate()`→`reconstruct()`→`postprocess()`→`export()`
chain: it reconstructs and writes one chunk of consecutive rows
(slices) at a time, via `StreamingVolumeWriter`
(`reconstruction/export.py`), so only one chunk's raw/normalized/
reconstructed data is ever in memory -- never the whole volume. Every
caller that already goes through `run_all()` -- `tomo run`, `tomo
batch`/`run_batch()`, and the GUI's "Reconstruct selected" (its
"Stream to disk" checkbox just sets this same field) -- gets this for
free, with no code of its own.

```python
config = ProjectConfig(
    path="scan.h5", format="hdf5", storage_mode="disk",
    center=128.0, tilt_deg=0.0,               # must already be resolved -- see below
    stream=True, stream_chunk_rows=None,       # None = size automatically
    export_format="hdf5", export_path="out/volume.h5",
)
result = PipelineRun(config).run_all()
```

Chunk size: `stream_chunk_rows`, or, if left `None`,
`orchestration.ram_check.estimate_chunk_rows(width, max_rows)`. Every
registered backend parallelizes across the slices within one
`reconstruct()` call internally, so chunk size is sized to keep every
CPU core busy in one call, unless that many rows wouldn't fit in
available RAM or there aren't that many rows left -- in which case
RAM/the remaining row count wins.

Two things must already be resolved before streaming can run;
`_run_streaming()` raises `PipelineStageError` otherwise:

- `center`: no default exists. Resolve it first (interactively, or set
  it directly).
- `dtype_clip_min`/`dtype_clip_max`, only if `dtype == "uint16"`: set
  both explicitly, or export as `float32`.

`tilt_deg` left `None` is treated as `0.0` rather than raising.

Chunks may be written in any order. For `format="hdf5"`, the dataset
is pre-allocated at its final `(total_slices, height, width)` shape
upfront, and each chunk is written by absolute index
(`dataset[z0:z1] = chunk`). For `format="tiff"`, each slice's filename
(`slice_00042.tif`) carries its own absolute index. `_run_streaming()`
writes chunks in increasing order, but nothing requires that.

A crop's z-range (`config.crop`'s first tuple) is applied as a write
filter: chunks, and the center/tilt math, still cover `config.rows`'
full range, and `postprocess()`'s non-streaming path also crops after
reconstructing that same full range -- only which rows get *written*
differs between the two paths. A per-row center, for a nonzero tilt,
is computed per chunk (`centers_for_rows()`, using row positions
relative to `config.rows`' own range) the same way the eager path's
`calibrate()` computes it for the whole range at once.

## Rotation

`reconstruct_rotated()` is `reconstruct()`, rotated by `rotate_deg`
degrees about the scan axis (every xy slice, uniformly):

```python
from reconstruction import Geometry, reconstruct_rotated

volume = reconstruct_rotated(sinogram, angles, center, geometry, backend="algotom_fbp", rotate_deg=30.0)
```

Same call shape and parameters as `reconstruct()`, plus `rotate_deg`.
One of two paths runs, chosen automatically per call:

- **Sinogram reorder** (`reconstruction/rotation.py`'s
  `shift_sinogram()`): relabels the sinogram's angle axis instead of
  resampling the reconstructed image. Used whenever
  `supports_exact_rotation(angles, geometry.geometry_type)` is `True`:
  `angles` evenly spaced, and its span covering either a full
  parallel-beam half-period (`>= 180deg`, via the identity
  `p(theta+180, s) = p(theta, -s)`), or a full period (`>= 360deg` --
  required for `geometry_type="cone"`, since that identity is
  parallel-beam-only). `rotate_deg` need not land on the angular
  sampling grid: an off-grid value is linearly interpolated along the
  angle axis. No built-in backend currently supports `"cone"` (see
  "Built-in backends" above), so this path's cone-beam branch is only
  exercised in tests, against a fake backend.
- **Image-domain fallback**: reconstructs normally, then rotates the
  volume with `scipy.ndimage.rotate` (bilinear, `reshape=False` --
  pixels rotated outside the original frame are lost). Used whenever
  `supports_exact_rotation()` is `False`, e.g. a limited-angle scan
  (angular span short of 180/360deg) or non-evenly-spaced angles.

`supports_exact_rotation(angles, geometry_type)` is public, for
checking which path a given scan will take without calling
`reconstruct_rotated()`.

`ProjectConfig.rotate_deg` (`None` = no rotation) drives this in
`PipelineRun.reconstruct()` and the streaming per-chunk loop
identically -- streaming rotates each chunk's own rows independently,
since rotation is a per-slice operation with no dependency between
rows. `PipelineRun` logs which path ran
(`"rotated {deg}deg via sinogram reorder"` / `"... via image-domain
fallback"`) and records `rotate_deg` in `run.settings`.

## Writing a backend

```python
from reconstruction.backend import ReconstructionBackend, register_backend

@register_backend
class MyBackend(ReconstructionBackend):
    name = "my_backend"
    supported_geometries = {"parallel"}  # or {"parallel", "cone"}
    requires = ("my_library",)           # modules it imports; missing -> not offered in the GUI

    def reconstruct(self, sinograms, angles, center, geometry, device=None, **options):
        ...  # sinograms: (n_angles, n_slices, width)
        return images  # (n_slices, height, width)
```

The contract specifies only shapes in and out; how a backend processes
the batch (looping per slice, a library's native batch API, CPU or
GPU) is internal to the backend.

Add an import line to `reconstruction/backends/__init__.py` to
register a new module. The GUI's algorithm list is built from the
registry (`reconstruction.list_backends()`), so a registered backend
whose `requires` are installed shows up there without editing the GUI.

Inputs are normalized transmission values (what `preprocessing`
returns), not attenuation: take the -log inside the backend (the
algotom and gridrec backends do). `center` is a scalar or one value per
slice; `device` is a hint -- report the truth in `actual_device()`.

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
