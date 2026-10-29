# Cone-beam reconstruction (FDK)

For laboratory micro-CT and any scan where the X-rays fan out from a point
source: the detector sees a cone of rays, not parallel ones. The `rtk_fdk`
backend reconstructs a **circular scan with a flat detector** with the
Feldkamp-Davis-Kress algorithm, on the CPU (no GPU needed), and the `astra_fdk` backend does the same on an NVIDIA GPU.

## What FDK does

Each projection is weighted by the cosine of its ray angle, every detector row
is ramp-filtered, and the result is backprojected along the cone rays into a
3D volume. It is exact in the central plane (the one containing the source) and
an approximation away from it: the farther a slice is from the centre, the larger
the cone angle and the stronger the "cone-beam artifacts" (blur and intensity
drift near the top and bottom of tall samples). This is the standard, fast
method; exact methods need a different scan path.

## Where it comes from

Writing and verifying our own cone-beam backprojector (and an OpenCL/CUDA
version of it) is a large job, and a well-tested implementation exists, so
the backend wraps the **Reconstruction Toolkit (RTK)**, Apache-2.0, which has a
multi-threaded CPU FDK. Install it with the `cone` extra:

```bash
pip install -e ".[cone]"          # itk-rtk; large, because it brings ITK
```

For the GPU, the `astra_fdk` backend wraps ASTRA's `FDK_CUDA` (`pip install astra-toolbox`; the wheel
brings its own CUDA runtime, so only an NVIDIA driver is needed). It uses the same geometry, conventions
and options as `rtk_fdk` and returns the same result (within a few percent: the two use slightly different
filter windows); it is listed as unavailable on a machine with no CUDA device. Both projections and volume
must fit in GPU memory; if they do not, use `size_xy` / `size_z` or `rtk_fdk`. `astra_fdk` is the
default for cone-beam data whenever it is usable (ASTRA installed and a CUDA device present); otherwise `rtk_fdk`,
the CPU one, is. Both are listed in the algorithm list either way (an unusable one greyed out).

## Geometry and conventions

The cone-beam generalization of the parallel-beam convention used everywhere else:

- The rotation axis is vertical; detector **rows run along the axis**, so the slices
  of the result are slices along the axis.
- At angle `theta` the detector's column axis is `u = (cos theta, -sin theta)` in the horizontal
  plane and the ray direction is `r = (sin theta, cos theta)`; the source is at `-SOD * r`, the detector
  plane at `(SDD - SOD) * r`. (SOD: source to rotation axis, SDD: source to detector.)
- `center` is the detector column onto which the rotation axis projects. The usual center-of-rotation
  finder (on the middle row's sinogram) works for it.
- Output: `(slices, y, x)`, cubic voxels, the size a detector pixel has at the axis,
  `pixel / (SDD / SOD)`; by default as many voxels across as the detector has columns and as many slices as
  detector rows (options `size_xy`, `size_z`).
- Values: attenuation coefficients, in 1/cm after the usual unit conversion.

These conventions were checked against analytic phantoms (two spheres at off-centre
positions along all three axes, with an off-centre axis): positions agree within a voxel and
the attenuation values within 10% (`tests/test_cone_beam.py`).

## What you need to provide

| What | Where |
|---|---|
| Geometry type = cone | Basic settings, or `completion_answers = {"geometry_type": "cone"}` |
| Source-detector distance, source-object distance (mm) | Basic settings (shown for cone beam), or the completion answers `source_to_detector_distance`, `source_to_object_distance` |
| **Pixel size at the sample** (um) | Basic settings' "Pixel size" (the `voxel_size` field) -- *not* the detector's own pitch. The detector pitch used in the reconstruction is `voxel_size * SDD / SOD`, unless you give `detector_pixel_size` yourself (then that wins). |
| Center of rotation | typed, or found with the Automatic adjustments tab's center scan (see `GUI_ARCHITECTURE.md`); the image-based metrics on the central plane are exact for cone beam, the sinogram ones approximate |

## Differences from parallel-beam runs

- **The whole detector at once.** Every output slice needs every detector row, so there is **no streaming**
  to disk and no restricting `rows` (use `crop` to keep part of the volume). Use "Reconstruct all slices"; the
  RAM check applies as usual.
- **The one single-slice preview is the central plane.** "Reconstruct (slice)" and the center-of-rotation scan
  reconstruct the plane through the source, perpendicular to the axis: its rays stay in it, so the middle detector
  row alone (the mean of the two middle rows for an even count) gives it exactly, as an FDK slice (`size_z=1`).
  Other slices need many rows and are only available in a full reconstruction.
- **One center of rotation**; a rotation-axis tilt correction is refused for cone-beam data.
- The algorithm list only offers backends that support the project's geometry: switching the geometry
  between parallel and cone changes the list (and the selected algorithm if it does not support
  the new geometry).

## Speed

CPU, 8 cores, all measured here: 180 projections of 64x96 -> 64x96x96 volume in 0.9 s; 360 of 128x128 ->
128^3 in 4.4 s; 360 of 256x256 -> 256^3 in 20 s (about 80 ms per slice).

GPU (`astra_fdk`, RTX 6000 Ada), 360 projections of 256x256 -> 256^3: 0.4 s, against 31 s for `rtk_fdk`.
Both are checked against a 3D ellipsoid phantom by `benchmarks/gpu_cone_phantom.py`.

## Not covered yet

Curved detectors, helical or laminography trajectories (tilted axis), detector/source offsets
other than the axis column, a multi-GPU or out-of-GPU-memory FDK, iterative cone-beam methods.
