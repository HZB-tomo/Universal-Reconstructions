# Cone-beam reconstruction (FDK)

For laboratory micro-CT and any scan where the X-rays fan out from a point
source: the detector sees a cone of rays, not parallel ones. The `rtk_fdk`
backend reconstructs a **circular scan with a flat detector** with the
Feldkamp-Davis-Kress algorithm, on the CPU (no GPU needed).

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

RTK also has CUDA versions, but this backend uses only the CPU one, so it does not use a GPU. A GPU
version could be added later behind the same backend contract.

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
| Center of rotation | as always (Find center, or typed) |

## Differences from parallel-beam runs

- **The whole detector at once.** Every output slice needs every detector row, so there is **no streaming**
  to disk, no restricting `rows` (use `crop` to keep part of the volume), and **no single-slice preview**:
  "Reconstruct (slice)" says so. Use "Reconstruct all slices"; the RAM check applies as usual.
- **One center of rotation**; a rotation-axis tilt correction is refused for cone-beam data.
- The algorithm list only offers backends that support the project's geometry: switching the geometry
  between parallel and cone changes the list (and the selected algorithm if it does not support
  the new geometry).

## Speed

CPU, 8 cores, all measured here: 180 projections of 64x96 -> 64x96x96 volume in 0.9 s; 360 of 128x128 ->
128^3 in 4.4 s; 360 of 256x256 -> 256^3 in 20 s (about 80 ms per slice).

## Not covered yet

Curved detectors, helical or laminography trajectories (tilted axis), detector/source offsets
other than the axis column, a GPU implementation, iterative cone-beam methods.
