"""
Center-of-rotation metrics, and scan_centers(): evaluate one metric at a grid of trial centers.

Where find_center()/search_center() (calibration/api.py) return the answer of one method, this module
returns the whole curve -- the metric at every trial center -- so a person can look at it, pick a center
by eye, and see how clear the minimum is. Every metric is normalized for plotting in the same way
(CenterScan.quality: 1 is the best trial, 0 the worst), whichever way its raw value points.

Two kinds of metric:

  "sinogram"  looks only at the projections of one row (cheap: milliseconds per trial center):
                pair_registration  projections 180 degrees apart must be mirror images about the axis
                vo                 symmetry of the sinogram in the Fourier domain (Vo et al. 2014)
  "slice"     reconstructs the slice at every trial center and scores the picture (one reconstruction
              per trial; the caller supplies `reconstruct(center) -> 2D image`):
                sharpness          gradient energy, high when edges are sharp
                total_variation    mean gradient magnitude, low for a clean picture
                entropy            histogram entropy, low for a picture of a few distinct materials
                negativity         energy of negative values, which a wrong center creates
                std                standard deviation (a weak criterion: depends on the sample)

For cone beam the slice is the central one (exact), and the sinogram metrics see the central row, a
fan-beam sinogram: only approximately mirror-symmetric, so "approximate" is set on the result.

Conventions: a center is a detector column (the rotation axis between columns 63 and 64 of a 128 wide
detector is 63.5); the "shift" of a center is its distance from the middle of the detector, (width - 1) / 2.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from calibration.api import half_scan_slice

Progress = Callable[[int, int], None]


class ScanCancelled(Exception):
    """Raised by scan_centers() when `cancelled()` turned true."""


def image_center(width: int) -> float:
    """The detector column in the middle of a `width` wide detector."""
    return (width - 1) / 2.0


def center_grid(width: int, shift_from: float, shift_to: float, points: int) -> np.ndarray:
    """`points` trial centers evenly spread over image_center(width) + [shift_from, shift_to]."""
    if points < 2:
        raise ValueError("a scan needs at least 2 points")
    if shift_to <= shift_from:
        raise ValueError(f"the scan range is empty: {shift_from} to {shift_to}")
    return image_center(width) + np.linspace(shift_from, shift_to, int(points))


# -- result ------------------------------------------------------------------------------------------------

@dataclass
class CenterScan:
    """
    The curve of one metric over a grid of trial centers.

    centers, values: the trial centers and the raw metric at each (NaN where it cannot be evaluated).
    quality: values rescaled to [0, 1], 1 = the best trial, whichever way the metric points.
    best: the best center, refined below the grid spacing by a parabola through the best trial and its
        neighbours (the best trial itself when it sits on the edge of the grid).
    at_edge: the best trial is the first or last: the true center may lie outside the range.
    rival: (center, quality) of the best local maximum of `quality` that is not part of the best's own
        peak, or None. A rival close to 1 means the curve is ambiguous.
    approximate: a note when the metric is only an approximation for this data (else "").
    """

    metric: str
    lower_is_better: bool
    centers: np.ndarray
    values: np.ndarray
    quality: np.ndarray
    best: float
    best_index: int
    at_edge: bool
    rival: Optional[tuple] = None
    approximate: str = ""

    def quality_at(self, center: float) -> float:
        """The quality of the curve at `center` (linear interpolation)."""
        return float(np.interp(center, self.centers, self.quality))


def _quality(values: np.ndarray, lower_is_better: bool) -> np.ndarray:
    v = np.array(values, dtype=float)
    finite = np.isfinite(v)
    if not finite.any():
        return np.zeros_like(v)
    lo, hi = v[finite].min(), v[finite].max()
    if hi - lo <= 0:
        return np.zeros_like(v)
    q = (hi - v) / (hi - lo) if lower_is_better else (v - lo) / (hi - lo)
    q[~finite] = 0.0
    return q


def _refine(centers: np.ndarray, quality: np.ndarray, i: int) -> float:
    if i == 0 or i == len(centers) - 1:
        return float(centers[i])
    y0, y1, y2 = quality[i - 1], quality[i], quality[i + 1]
    denominator = y0 - 2.0 * y1 + y2
    if denominator >= 0:         # not a peak
        return float(centers[i])
    offset = 0.5 * (y0 - y2) / denominator
    return float(centers[i] + np.clip(offset, -1.0, 1.0) * (centers[1] - centers[0]))


def _rival(centers: np.ndarray, quality: np.ndarray, best: int) -> Optional[tuple]:
    """The best point outside the best peak: walk down both sides of the best until the curve turns up."""
    n = len(quality)
    lo = best
    while lo > 0 and quality[lo - 1] <= quality[lo]:
        lo -= 1
    hi = best
    while hi < n - 1 and quality[hi + 1] <= quality[hi]:
        hi += 1
    far = 0.08 * (centers[-1] - centers[0])          # a ripple next to the peak is not a second candidate
    outside = [j for j in range(n) if (j < lo or j > hi) and abs(centers[j] - centers[best]) >= far]
    if not outside:
        return None
    j = max(outside, key=lambda k: quality[k])
    # only a rival if it is a local maximum of its own, not just the shoulder of the best peak
    left = quality[j - 1] if j > 0 else -1.0
    right = quality[j + 1] if j < n - 1 else -1.0
    if quality[j] < left or quality[j] < right or quality[j] <= 0.0:
        return None
    return float(centers[j]), float(quality[j])


# -- metrics -----------------------------------------------------------------------------------------------

class CenterMetric(ABC):
    name: str
    label: str
    kind: str                 # "sinogram" | "slice"
    lower_is_better: bool
    description: str = ""

    @abstractmethod
    def curve(
        self, centers: np.ndarray, *, sinogram: Optional[np.ndarray] = None, angles: Optional[np.ndarray] = None,
        reconstruct: Optional[Callable[[float], np.ndarray]] = None, progress: Optional[Progress] = None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> np.ndarray:
        """The raw metric at every trial center. `sinogram` is (n_angles, width) normalized transmission,
        `angles` in degrees; slice metrics call reconstruct(center) instead."""


_REGISTRY: dict = {}


def register_metric(cls):
    instance = cls()
    _REGISTRY[instance.name] = instance
    return cls


def get_metric(name: str) -> CenterMetric:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown center-of-rotation metric '{name}'. Registered: {list(_REGISTRY)}")
    return _REGISTRY[name]


# The reliable ones first; the noise-sensitive ones last.
_DISPLAY_ORDER = ("vo", "pair_registration", "sharpness", "entropy", "total_variation", "negativity", "std")


def list_metrics() -> list:
    """Registered metric names, in display order."""
    known = [n for n in _DISPLAY_ORDER if n in _REGISTRY]
    return known + [n for n in _REGISTRY if n not in known]


def _need_sinogram(sinogram, angles):
    if sinogram is None or angles is None:
        raise ValueError("this metric needs the sinogram and its angles")
    sinogram = np.asarray(sinogram, dtype=np.float32)
    if sinogram.ndim != 2 or sinogram.shape[0] != len(angles):
        raise ValueError(f"expected a (n_angles, width) sinogram with {len(angles)} angles, got {sinogram.shape}")
    return sinogram, np.asarray(angles, dtype=float)


@register_metric
class PairRegistration(CenterMetric):
    name = "pair_registration"
    label = "0/180 projection pairs"
    kind = "sinogram"
    lower_is_better = True
    description = (
        "Two projections 180 degrees apart are mirror images of each other about the rotation axis. For each "
        "trial center the later one is mirrored about it and compared with the earlier one; the center with the "
        "smallest mismatch wins. Needs projections 180 degrees apart (a 0-180 scan gives only a few pairs; a "
        "0-360 scan gives many, which makes it more robust)."
    )
    MAX_PAIRS = 64
    MIN_OVERLAP = 0.3

    @staticmethod
    def pairs(angles: np.ndarray) -> list:
        """(i, j) index pairs whose angles differ by 180 degrees, each pair once. Exact partners (within half an
        angular step) when the scan has them; a scan that stops short of 180 degrees past its first projection
        (0 to 179.5 in half degrees) gets the nearest partner within 1.5 steps instead, which is nearly as good."""
        n = len(angles)
        if n < 2:
            return []
        step = float(np.median(np.abs(np.diff(np.sort(angles)))))
        for tolerance in (0.51 * step, 1.5 * step):
            found = {}
            for i in range(n):
                j = int(np.argmin(np.abs(angles - (angles[i] + 180.0))))
                if abs(angles[j] - angles[i] - 180.0) <= max(tolerance, 1e-6):
                    found[(i, j)] = True
            if found:
                return list(found)
        return []

    def curve(self, centers, *, sinogram=None, angles=None, reconstruct=None, progress=None, cancelled=lambda: False):
        sinogram, angles = _need_sinogram(sinogram, angles)
        pairs = self.pairs(angles)
        if not pairs:
            raise ValueError(
                "this method needs projections 180 degrees apart, and this scan has none "
                f"(angles {angles.min():.1f} to {angles.max():.1f}); use the sinogram symmetry (Vo) method"
            )
        if len(pairs) > self.MAX_PAIRS:
            pairs = [pairs[k] for k in np.linspace(0, len(pairs) - 1, self.MAX_PAIRS).astype(int)]
        attenuation = -np.log(np.clip(sinogram, 1e-6, None))
        first = attenuation[[i for i, _ in pairs]].astype(np.float64)
        second = attenuation[[j for _, j in pairs]].astype(np.float64)
        width = sinogram.shape[1]
        u = np.arange(width, dtype=np.float64)
        out = np.full(len(centers), np.nan)
        for k, c in enumerate(centers):
            if cancelled():
                raise ScanCancelled()
            x = 2.0 * c - u                                  # the column of `first` that mirrors column u of `second`
            valid = (x >= 0.0) & (x <= width - 1.0)
            if valid.sum() >= self.MIN_OVERLAP * width:
                xv = x[valid]
                x0 = np.minimum(np.floor(xv).astype(int), width - 2)
                f = xv - x0
                mirrored = first[:, x0] * (1.0 - f) + first[:, x0 + 1] * f
                reference = second[:, valid]
                out[k] = ((mirrored - reference) ** 2).sum() / (np.square(reference).sum() + 1e-12)
            if progress:
                progress(k + 1, len(centers))
        return out


@register_metric
class VoSymmetry(CenterMetric):
    name = "vo"
    label = "Sinogram symmetry (Vo)"
    kind = "sinogram"
    lower_is_better = True
    description = (
        "Vo's method: a sinogram of 180 degrees joined to its mirror image about the trial center has a "
        "particular structure in the Fourier domain exactly when the center is right. Robust to noise and "
        "to stripes, and works on a 0-180 scan."
    )
    RATIO = 0.5

    def curve(self, centers, *, sinogram=None, angles=None, reconstruct=None, progress=None, cancelled=lambda: False):
        sinogram, angles = _need_sinogram(sinogram, angles)
        try:
            import algotom.prep.calculation as calc
        except ImportError as e:
            raise ImportError("the 'vo' metric requires the 'algotom' package. Install it with: pip install algotom") from e
        from concurrent.futures import ThreadPoolExecutor

        from scipy import ndimage as ndi

        half = sinogram[half_scan_slice(angles)]
        nrow, ncol = half.shape
        smooth = ndi.gaussian_filter(half, (2, 2), mode="reflect")
        flip, comp = np.fliplr(smooth), np.flipud(smooth)
        mask = calc.make_inverse_double_wedge_mask(2 * nrow, ncol, 0.5 * self.RATIO * ncol, None, None)
        done = [0]

        def one(c):
            if cancelled():
                return float("nan")
            value = float(calc.calculate_center_metric(float(c), smooth, flip, comp, mask))
            done[0] += 1
            if progress:
                progress(done[0], len(centers))
            return value

        with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
            out = np.array(list(pool.map(one, centers)), dtype=float)
        if cancelled():
            raise ScanCancelled()
        return out


def fov_mask(shape: tuple) -> np.ndarray:
    """Boolean disc covering 95% of the smaller side of an image: the part that is inside the field of view."""
    h, w = shape
    y, x = np.mgrid[:h, :w]
    return (y - (h - 1) / 2.0) ** 2 + (x - (w - 1) / 2.0) ** 2 <= (0.475 * min(h, w)) ** 2


class _SliceMetric(CenterMetric):
    kind = "slice"

    @abstractmethod
    def score(self, image: np.ndarray, mask: np.ndarray) -> float:
        ...

    def curve(self, centers, *, sinogram=None, angles=None, reconstruct=None, progress=None, cancelled=lambda: False):
        if reconstruct is None:
            raise ValueError("this metric reconstructs the slice at every trial center: it needs reconstruct(center)")
        out = np.full(len(centers), np.nan)
        mask = None
        for k, c in enumerate(centers):
            if cancelled():
                raise ScanCancelled()
            image = np.asarray(reconstruct(float(c)), dtype=np.float64)
            if mask is None or mask.shape != image.shape:
                mask = fov_mask(image.shape)
            out[k] = self.score(image, mask)
            if progress:
                progress(k + 1, len(centers))
        return out


def _gradient(image: np.ndarray):
    gy, gx = np.gradient(image)
    return gy, gx


@register_metric
class Sharpness(_SliceMetric):
    name = "sharpness"
    label = "Sharpness (gradient energy)"
    lower_is_better = False
    description = "Mean squared gradient inside the field of view. A wrong center smears every edge, which lowers it."

    def score(self, image, mask):
        gy, gx = _gradient(image)
        return float(np.mean((gx ** 2 + gy ** 2)[mask]))


@register_metric
class TotalVariation(_SliceMetric):
    name = "total_variation"
    label = "Total variation"
    lower_is_better = True
    description = (
        "Mean gradient magnitude inside the field of view. A wrong center adds arcs and streaks, which raise it. "
        "Sensitive to noise: on noisy data the minimum can be broad or off."
    )

    def score(self, image, mask):
        gy, gx = _gradient(image)
        return float(np.mean(np.hypot(gx, gy)[mask]))


@register_metric
class Entropy(_SliceMetric):
    name = "entropy"
    label = "Entropy"
    lower_is_better = True
    description = (
        "Entropy of the histogram of the picture. A sharp picture of a few distinct materials has a few "
        "narrow peaks, a smeared one a broad spread."
    )
    BINS = 256

    def score(self, image, mask):
        values = image[mask]
        lo, hi = np.percentile(values, (0.5, 99.5))
        if hi <= lo:
            return 0.0
        counts, _ = np.histogram(values, bins=self.BINS, range=(lo, hi))
        p = counts[counts > 0] / counts.sum()
        return float(-(p * np.log2(p)).sum())


@register_metric
class Negativity(_SliceMetric):
    name = "negativity"
    label = "Negativity"
    lower_is_better = True
    description = (
        "Share of the picture's energy that is negative. Attenuation cannot be negative, but a wrong center "
        "creates dark arcs and rims. Needs a sample on an empty background, and is sensitive to noise and to "
        "the filter's own undershoot: use it as a cross-check."
    )

    def score(self, image, mask):
        values = image[mask]
        return float(np.square(np.minimum(values, 0.0)).sum() / (np.square(values).sum() + 1e-12))


@register_metric
class Std(_SliceMetric):
    name = "std"
    label = "Standard deviation (weak)"
    lower_is_better = False
    description = (
        "Standard deviation of the picture. A smeared picture usually has less contrast, but it also depends "
        "on the sample, so use it only as a cross-check of the other metrics."
    )

    def score(self, image, mask):
        return float(np.std(image[mask]))


# -- the scan ----------------------------------------------------------------------------------------------

def scan_centers(
    metric: str, centers, *, sinogram: Optional[np.ndarray] = None, angles: Optional[np.ndarray] = None,
    reconstruct: Optional[Callable[[float], np.ndarray]] = None, progress: Optional[Progress] = None,
    cancelled: Callable[[], bool] = lambda: False, approximate: str = "",
) -> CenterScan:
    """
    Evaluate `metric` (a registered name, see list_metrics()) at every trial center and return the curve
    with its best center. Raises ScanCancelled if `cancelled()` turns true; ValueError for input the
    metric cannot use. `approximate` is carried to the result for the caller to show.
    """
    m = get_metric(metric)
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or len(centers) < 2:
        raise ValueError("a scan needs at least 2 trial centers")
    values = m.curve(
        centers, sinogram=sinogram, angles=angles, reconstruct=reconstruct, progress=progress, cancelled=cancelled,
    )
    return scan_from_curve(metric, centers, values, approximate=approximate)


def scan_from_curve(
    metric: str, centers, values, best: Optional[float] = None, approximate: str = "",
) -> CenterScan:
    """
    A CenterScan from a curve that was already computed (for example the one a search evaluated on its way),
    so it can be shown without computing anything again. `best` overrides the center the curve's own minimum
    gives, for when a finer method has a better answer.
    """
    m = get_metric(metric)
    centers = np.asarray(centers, dtype=float)
    values = np.asarray(values, dtype=float)
    quality = _quality(values, m.lower_is_better)
    index = int(np.argmax(quality))
    return CenterScan(
        metric=m.name, lower_is_better=m.lower_is_better, centers=centers, values=values, quality=quality,
        best=_refine(centers, quality, index) if best is None else float(best), best_index=index,
        at_edge=index in (0, len(centers) - 1), rival=_rival(centers, quality, index), approximate=approximate,
    )
