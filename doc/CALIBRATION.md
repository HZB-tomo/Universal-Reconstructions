# Calibration

`calibration/` finds center-of-rotation and rotation-axis tilt from
normalized sinograms/projections. Structure: a backend contract and
registry (`backend.py`), one built-in backend (`backends/algotom_cor.py`),
and a backend-agnostic public API (`api.py`).

Three separate functions:

| Function | Finds | Rows needed |
|---|---|---|
| `find_center()` | center of rotation | one |
| `find_tilt()` | rotation-axis tilt | several |
| `centers_for_rows()` | per-row centers from a tilt and a center | none (pure arithmetic) |

`find_center()` and `find_tilt()` can be replaced by a manually
determined number. `centers_for_rows()` takes plain numbers
(`tilt_deg`, `center`, `reference_row`), not a result object tied to
one specific function, so a manually determined center and an
automatically found tilt can be combined directly.

If `algotom` is unavailable, a replacement backend only needs to
implement `CalibrationBackend.find_center()`; nothing in `api.py`
changes.

## `half_scan_slice()`

```python
from calibration import half_scan_slice

half = half_scan_slice(angles)
center = find_center(normalized[half, middle_row, :], backend="algotom")
result = find_tilt(normalized[half], backend="algotom")
```

Returns the slice along the angle axis that covers the first 180
degrees of a scan. For a scan of 180 degrees or less, or when `angles`
is `None`, it returns `slice(None)`. `PipelineRun.calibrate()` and the
GUI use it for center and tilt finding.

## `find_center()`

```python
from calibration import find_center

center = find_center(sino_180, backend="algotom")
```

Finds the center of rotation of a single 180-degree sinogram,
`(n_angles, width)`, typically the stack's middle row,
`normalized[:, n_rows // 2, :]`.

`start`/`stop` bound the search range in column-index units. Defaults
to `width/2 +/- width/4` if omitted. Pass `start`/`stop` explicitly for
offsets larger than a quarter of the width.

`backend` is a registered backend name or a `CalibrationBackend`
instance.

## `find_tilt()`

```python
from calibration import find_tilt

result = find_tilt(normalized, backend="algotom", num_rows=12, margin=4)
result.tilt_deg                # degrees
result.center_at_reference_row # center of rotation at the middle row
```

`normalized` is a full projection stack, `(n_angles, n_rows, width)`,
angle-first, the same shape `preprocessing.normalize_rows()` returns.

Method:

1. Sample `num_rows` rows spread evenly across the stack, skipping
   `margin` rows at the top and bottom.
2. Call `find_center()` independently on each sampled row's sinogram.
3. Drop any row whose estimate is more than `outlier_tolerance` pixels
   (default 20) from the median of all estimates.
4. Fit a line (center vs. row index) through the remaining rows.
5. The slope gives `tilt_deg = degrees(arctan(slope))`. The line's
   value at the stack's middle row is returned as
   `center_at_reference_row`.

`center`, if given, is an already-known center of rotation (typically
from `find_center()` at the stack's middle row). Each row's search
window is re-centered on `center +/- width/4` instead of the
sinogram's nominal midpoint. This matters only when the true offset
falls outside the default window; at an offset within a real
detector's typical range it has no measurable effect.

At an offset large enough to fall outside the search window on its
own, several rows can lock onto a second, competing minimum roughly as
strong as the true one. When that happens the per-row estimates split
into two clusters and `find_tilt()` raises `RuntimeError` rather than
fitting a line through a mix of both.

**"Tilt" here means rotation-axis tilt (axis lean)**: the physical
rotation axis is not parallel to the detector's row axis, so the
effective center of rotation drifts linearly with row. This is
distinct from an in-plane rotation of each projection frame (detector
roll), which `find_tilt()` does not detect; roll mixes row and column
information, breaking the per-row independence this method relies on.
`tests/fixtures/realistic_phantom`'s `tilt_deg` parameter simulates
axis lean; see [`REALISTIC_PHANTOM.md`](./REALISTIC_PHANTOM.md) and
`simulate.py`'s `apply_cor_and_tilt()`.

## `centers_for_rows()`

```python
from calibration import centers_for_rows

result = find_tilt(normalized, backend="algotom")
center = find_center(normalized[:, middle_row, :], backend="algotom")

centers = centers_for_rows(result.tilt_deg, center, reference_row=middle_row, rows=[6, 9, 12, 15, 18])
images = reconstruct(normalized[:, [6,9,12,15,18], :], angles, centers, geometry, backend="algotom_fbp")
```

Builds a per-row center array from a tilt angle and a center at one
known row, ready to pass to `reconstruction.reconstruct()`'s `center`
parameter. `tilt_deg`, `center`, and `reference_row` may come from
`find_tilt()`, `find_center()`, or a value supplied directly.

`rows` is the row indices to compute centers for; it does not need to
match the rows `find_tilt()` sampled, or include `reference_row`.

No pixel-level transform is applied to the projection data; rotation-
axis tilt is corrected entirely by using a different center per row.

## `algotom` backend

Wraps `algotom.prep.calculation.find_center_vo`:

> Vo, N.T., Drakopoulos, M., Atwood, R.C., Reinhard, C., "Reliable
> method for calculating the center of rotation in parallel-beam
> tomography," *Optics Express* 22(16), 19078-19086 (2014).
> https://doi.org/10.1364/OE.22.019078

Requires the optional `algotom` package (`pip install .[preprocessing]`).
`get_backend("algotom")` resolves without it installed; calling
`.find_center()` requires the real library.

Related algotom methods, not currently wrapped:

- `find_center_based_phase_correlation` (0-degree/180-degree projection
  image pair): Argyriou, V., Vlachos, T., "Estimation of sub-pixel
  motion using gradient cross-correlation," *Electronics Letters*
  39(13), 980-982 (2003). https://doi.org/10.1049/el:20030666
- `find_center_360` (offset-axis / extended-field-of-view scans): Vo,
  N.T., Atwood, R.C., Drakopoulos, M., Connolley, T., "Data processing
  methods and data acquisition for samples larger than the field of
  view in parallel-beam tomography," *Optics Express* 29(12),
  17849-17874 (2021). https://doi.org/10.1364/OE.418448. Requires a
  sinogram-stitching step
  (`algotom.prep.conversion.convert_sinogram_360_to_180`) before
  reconstruction.

## Writing a backend

```python
from calibration.backend import CalibrationBackend, register_backend

@register_backend
class MyBackend(CalibrationBackend):
    name = "my_backend"

    def find_center(self, sino_180, start=None, stop=None, **options):
        ...
        return center  # float
```

Add an import line to `calibration/backends/__init__.py` to register
the new module. `find_tilt()` requires no changes to use a new
backend.

## See also

- [`RECONSTRUCTION.md`](./RECONSTRUCTION.md): how a found center/tilt
  feeds into reconstruction.
- [`REALISTIC_PHANTOM.md`](./REALISTIC_PHANTOM.md): the test-data
  generator this module is tested against.
