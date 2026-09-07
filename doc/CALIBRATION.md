# Calibration

`calibration/` finds center-of-rotation and rotation-axis tilt from
normalized sinograms/projections. Same shape as `reconstruction/`, and
deliberately so: a backend contract + registry (`backend.py`), one
built-in backend (`backends/algotom_cor.py`), and a small, backend-
agnostic public API (`api.py`) that does the actual orchestration.

## Why it's structured like `reconstruction/`, not as plain functions

The one backend today, `algotom`, wraps a real published algorithm
(cited below) -- not a black box. If algotom is ever unavailable or
removed, a from-scratch reimplementation of that published algorithm
only has to satisfy `CalibrationBackend`'s one method, `find_center()`,
to slot in as a replacement. Nothing in `api.py`, or anything calling
it, would need to change. This is the same reasoning
`reconstruction/backend.py` uses for `ReconstructionBackend` -- explicit
selection, no auto-detection, and the interchangeable part kept to the
smallest real primitive.

## `find_center()`

```python
from calibration import find_center

center = find_center(sino_180, backend="algotom")
```

Finds the center of rotation of a single 180-degree sinogram,
`(n_angles, width)`. `start`/`stop` bound the search range in
column-index units; if omitted, defaults to `width/2 +/- width/4`.
That default matters more than it looks: algotom's own internal
default (`width/2 +/- width/16`) is confirmed too narrow for anything
beyond a small deviation, but widening it all the way to the sinogram's
full column range is worse, not safer -- confirmed directly, the
underlying search becomes markedly less reliable across that much
range, latching onto the wrong minimum even on otherwise-good rows.
Pass `start`/`stop` explicitly if your setup's real deviations exceed
a quarter of the width.

`backend` is always explicit -- a name (looked up via `get_backend()`)
or a `CalibrationBackend` instance directly, same convention as
`reconstruction.reconstruct()`.

## `find_tilt_and_center()`

```python
from calibration import find_tilt_and_center

result = find_tilt_and_center(normalized, backend="algotom", num_rows=12, margin=4)
result.tilt_deg   # degrees
result.center     # center of rotation at the middle row
```

This is the one that matters most in practice. `normalized` is a full
projection stack, `(n_angles, n_rows, width)` -- angle-first, exactly
what `preprocessing.normalize_rows()` already returns, so its result
can be handed to this function directly, no reshaping.

**Method**: sample `num_rows` rows spread across the stack (skipping
`margin` rows at the top/bottom), call `find_center()` independently on
each row's own sinogram, drop rows whose estimate is more than
`outlier_tolerance` pixels (default 20, matching real reference code)
from the median, then fit a line -- center vs. row index -- through
what's left. The slope, converted to an angle, is the tilt; the value
at the middle row is the reported center. This is the exact method
used by real reference code,
[`oldscripts/Reconstruction_algotom_loop.ipynb`](../oldscripts/Reconstruction_algotom_loop.ipynb)'s
own `find_tilt_and_center()` -- adopted here after checking that
reference, not designed from scratch.

**What "tilt" means here, precisely** -- this needed real verification,
not assumption, and got it wrong once along the way (see git history /
project memory): "tilt" is overloaded in casual CT usage, and two
genuinely different physical effects hide behind it:

- **Rotation-axis tilt (axis lean)** -- the physical rotation axis
  isn't parallel to the detector's row axis, so the effective center of
  rotation drifts linearly with row. This is what `find_tilt_and_center()`
  detects, and it's exact for this method: each row is a fully
  independent sinogram with its own true center, no approximation.
  Verified directly: a hand-constructed, known axis lean was recovered
  to within about 1% of its true slope/angle.
- **Detector roll** -- an in-plane rotation of each projection frame.
  This is a genuinely different, harder problem: a roll mixes row and
  column information (a fixed physical height smears across several
  detector rows depending on column position), so treating rows
  independently doesn't work. Verified directly: applying this exact
  method to a simulated 1-degree roll produced a recovered "tilt" of
  roughly **-43 degrees** -- not a small error, complete nonsense.
  Detector-roll detection/correction is explicitly out of scope here.

`tests/fixtures/realistic_phantom`'s `tilt_deg` parameter simulates
axis lean specifically for this reason -- see
[`REALISTIC_PHANTOM.md`](./REALISTIC_PHANTOM.md) and
`simulate.py`'s `apply_cor_and_tilt()` docstring. It used to simulate
detector roll; that was changed once this distinction was confirmed,
since a roll-based phantom could never validate this method correctly.

**Correcting axis lean needs no new code.** Since it's exactly "a
different, linearly-varying center per row," and
`reconstruction.reconstruct()` already accepts a per-slice center
array as a first-class feature, the fitted `center(row) = slope*row +
intercept` line can be turned into a per-row center array and handed
straight to `reconstruct()` -- more exact than the common practical
shortcut of approximating the correction with a whole-frame rotation
(what the reference notebook above does; a small-angle approximation
of the true shear, not an exact match).

**Why this method, and not a reconstruction-quality search**:
`find_center()`/`find_tilt_and_center()` never reconstruct anything --
pure sinogram-domain math, fully deterministic. An alternative design
(reconstruct several candidate centers/slices, score each with an
image-quality metric, pick the best) was investigated and set aside:
verified directly, `reconstruction.reconstruct()`'s CPU-fallback FBP
path isn't bit-reproducible between identical calls (~3% variance
difference call to call), which swamps the small differences such a
metric would need to detect near the true optimum -- tried against our
own phantom with image variance, gradient-sharpness, and the exact
Donath-entropy formula from `oldscripts/BAMline_recon_v0p2.py`'s own
`entropyMetric()`, none of which reliably localized the known true
center. A manual/visual check (reconstruct a range of candidates and
look at them) sidesteps this entirely -- a human doesn't care about 3%
noise -- and needs no function of its own here: it's just a direct call
to `reconstruction.reconstruct()` with an array of candidate centers,
exactly what `Reconstruction_algotom_loop.ipynb`'s own
`center_visual_scan()` does (also with no automatic scoring).

## `algotom` -- the built-in backend

Wraps `algotom.prep.calculation.find_center_vo`:

> Vo, N.T., Drakopoulos, M., Atwood, R.C., Reinhard, C., "Reliable
> method for calculating the center of rotation in parallel-beam
> tomography," *Optics Express* 22(16), 19078-19086 (2014).
> https://doi.org/10.1364/OE.22.019078

Needs the optional `algotom` package (`pip install .[preprocessing]`,
the same extra `preprocessing/corrections.py` and `algotom_fbp` use).
`get_backend("algotom")` resolves fine without it installed -- only
calling `.find_center()` needs the real library.

Other algotom methods considered, not currently wrapped, for future
reference:

- `find_center_based_phase_correlation` (0-degree/180-degree projection
  *image* pair, not a sinogram -- a fast rough estimate): Argyriou, V.,
  Vlachos, T., "Estimation of sub-pixel motion using gradient
  cross-correlation," *Electronics Letters* 39(13), 980-982 (2003).
  https://doi.org/10.1049/el:20030666
- `find_center_360` (offset-axis / extended-field-of-view scans, where
  a sample is larger than the detector's field of view): Vo, N.T.,
  Atwood, R.C., Drakopoulos, M., Connolley, T., "Data processing
  methods and data acquisition for samples larger than the field of
  view in parallel-beam tomography," *Optics Express* 29(12),
  17849-17874 (2021). https://doi.org/10.1364/OE.418448 -- needs a
  sinogram-stitching step (`algotom.prep.conversion.convert_sinogram_360_to_180`)
  before reconstruction can proceed at all; that's a data-shape
  transform closer to `preprocessing/` than to calibration, and neither
  piece is built yet.

## Writing a backend

Exactly one method:

```python
from calibration.backend import CalibrationBackend, register_backend

@register_backend
class MyBackend(CalibrationBackend):
    name = "my_backend"

    def find_center(self, sino_180, start=None, stop=None, **options):
        ...
        return center  # float
```

Register the new module the same way vendor plugins and reconstruction
backends do: add an import line to `calibration/backends/__init__.py`.
`find_tilt_and_center()` needs no changes to use a new backend -- it's
built entirely on `find_center()`.

## Where to go next

- [`oldscripts/Reconstruction_algotom_loop.ipynb`](../oldscripts/Reconstruction_algotom_loop.ipynb) --
  the real reference implementation this module's `find_tilt_and_center()`
  is modeled on.
- [`RECONSTRUCTION.md`](./RECONSTRUCTION.md) -- how a found center/tilt
  feeds into an actual reconstruction, including the per-row center
  array trick that corrects axis lean.
- [`REALISTIC_PHANTOM.md`](./REALISTIC_PHANTOM.md) -- the test-data
  generator this module is verified against.
