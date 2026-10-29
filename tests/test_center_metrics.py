"""calibration.metrics: the curve of a metric over trial centers of rotation, and the scan result built from it."""
import numpy as np
import pytest
from scipy.ndimage import rotate

import calibration
from calibration import ScanCancelled, center_grid, image_center, list_metrics, scan_centers
from calibration.metrics import CenterMetric, PairRegistration, register_metric
from reconstruction import Geometry, reconstruct

N = 192
TRUE = (N - 1) / 2.0 + 20      # the disks are rotated about the middle of the image, then rolled by 20 columns


@pytest.fixture(scope="module")
def phantom_sinogram():
    """Transmission sinogram of a few disks over 360 degrees (1 degree steps), rotation axis at column 115.5."""
    y, x = np.mgrid[:N, :N] - N / 2
    img = np.zeros((N, N), np.float32)
    for cx, cy, r, v in [(0, 0, .40, 1.0), (.1, .05, .15, -.5), (-.15, -.1, .10, .8), (.0, -.2, .06, 1.5)]:
        img[(x - cx * N) ** 2 + (y - cy * N) ** 2 < (r * N) ** 2] += v
    angles = np.arange(360, dtype=float)
    line = np.stack([rotate(img, -a, reshape=False, order=1).sum(axis=0) for a in angles]).astype(np.float32)
    attenuation = np.roll(line, 20, axis=1) * (3.0 / line.max()) * 0.1
    return np.exp(-attenuation), angles


def test_grid_and_image_center():
    assert image_center(128) == 63.5
    grid = center_grid(128, -2.0, 3.0, 11)
    assert grid[0] == 61.5 and grid[-1] == 66.5 and len(grid) == 11 and np.allclose(np.diff(grid), 0.5)
    with pytest.raises(ValueError, match="at least 2"):
        center_grid(128, -1, 1, 1)
    with pytest.raises(ValueError, match="empty"):
        center_grid(128, 2, -2, 10)


def test_every_metric_is_registered_in_display_order():
    builtin = ["vo", "pair_registration", "sharpness", "entropy", "total_variation", "negativity", "std"]
    assert list_metrics()[:len(builtin)] == builtin          # (other tests register made-up metrics after these)
    assert {calibration.get_metric(m).kind for m in list_metrics()} == {"sinogram", "slice"}
    with pytest.raises(KeyError, match="Unknown"):
        calibration.get_metric("nope")


class _Dips(CenterMetric):
    """A made-up metric (lower is better) with a deep dip at 40.3 and a shallower one at 70."""

    name = "test_dips"
    label = "dips"
    kind = "sinogram"
    lower_is_better = True

    def curve(self, centers, **_):
        return 1.0 - np.exp(-((centers - 40.3) / 3) ** 2) - 0.7 * np.exp(-((centers - 70) / 3) ** 2)


def test_quality_best_refinement_and_rival():
    register_metric(_Dips)
    scan = scan_centers("test_dips", np.arange(20.0, 90.0, 1.0))
    assert scan.quality.max() == 1.0 and scan.quality.min() == 0.0 and scan.quality[scan.best_index] == 1.0
    assert scan.best == pytest.approx(40.3, abs=0.1)              # finer than the 1 px grid
    assert not scan.at_edge
    assert scan.rival[0] == pytest.approx(70.0, abs=1.0) and scan.rival[1] == pytest.approx(0.7, abs=0.05)
    edge = scan_centers("test_dips", np.arange(40.3, 90.0, 1.0)[:20])      # best at the first trial
    assert edge.at_edge
    assert scan.quality_at(40.3) == pytest.approx(1.0, abs=0.02)


def test_higher_is_better_metrics_are_flipped_and_flat_curves_are_all_zero():
    class Up(CenterMetric):
        name, label, kind, lower_is_better = "test_up", "up", "sinogram", False

        def curve(self, centers, **_):
            return -(centers - 5.0) ** 2

    register_metric(Up)
    scan = scan_centers("test_up", np.linspace(0, 10, 21))
    assert scan.best == pytest.approx(5.0, abs=0.01) and scan.values.max() == 0.0

    class Flat(Up):
        name = "test_flat"

        def curve(self, centers, **_):
            return np.ones(len(centers))

    register_metric(Flat)
    assert not scan_centers("test_flat", np.linspace(0, 10, 5)).quality.any()


def test_progress_and_cancellation(phantom_sinogram):
    sino, angles = phantom_sinogram
    seen = []
    scan_centers("pair_registration", center_grid(N, 50, 60, 11), sinogram=sino, angles=angles,
                 progress=lambda done, total: seen.append((done, total)))
    assert seen[0] == (1, 11) and seen[-1] == (11, 11)
    with pytest.raises(ScanCancelled):
        scan_centers("pair_registration", center_grid(N, 50, 60, 11), sinogram=sino, angles=angles, cancelled=lambda: True)
    with pytest.raises(ScanCancelled):
        scan_centers("vo", center_grid(N, 50, 60, 11), sinogram=sino, angles=angles, cancelled=lambda: True)


@pytest.mark.parametrize("name", ["pair_registration", "vo"])
def test_sinogram_metrics_find_the_axis(phantom_sinogram, name):
    pytest.importorskip("algotom")
    sino, angles = phantom_sinogram
    scan = scan_centers(name, TRUE + np.linspace(-15, 15, 61), sinogram=sino, angles=angles)
    assert scan.best == pytest.approx(TRUE, abs=0.5)
    assert not scan.at_edge and scan.quality_at(TRUE + 10) < 0.95 * scan.quality_at(TRUE)


def test_a_half_scan_has_no_pairs_for_pair_registration_but_vo_works(phantom_sinogram):
    pytest.importorskip("algotom")
    sino, angles = phantom_sinogram
    half = slice(0, 90)                                          # 0-89 degrees: nothing 180 degrees apart
    with pytest.raises(ValueError, match="180 degrees apart"):
        scan_centers("pair_registration", center_grid(N, 40, 60, 11), sinogram=sino[half], angles=angles[half])
    assert PairRegistration.pairs(angles[:181])                  # 0 and 180 are both there
    assert len(PairRegistration.pairs(angles)) == 180


def test_sinogram_metrics_need_sinogram_and_angles():
    with pytest.raises(ValueError, match="needs the sinogram"):
        scan_centers("vo", np.arange(5.0))
    with pytest.raises(ValueError, match="needs reconstruct"):
        scan_centers("sharpness", np.arange(5.0))


@pytest.fixture(scope="module")
def reconstruction_at(phantom_sinogram):
    pytest.importorskip("algotom")
    sino, angles = phantom_sinogram

    def at(center):
        return reconstruct(sino[:180], angles[:180], center, geometry=Geometry("parallel"), backend="algotom_fbp", device="cpu")

    return at


@pytest.mark.parametrize("name, tolerance", [("sharpness", 1.5), ("entropy", 0.5)])
def test_reliable_slice_metrics_find_the_axis(reconstruction_at, name, tolerance):
    scan = scan_centers(name, TRUE + np.linspace(-6, 6, 25), reconstruct=lambda c: reconstruction_at(c))
    assert scan.best == pytest.approx(TRUE, abs=tolerance) and not scan.at_edge


@pytest.mark.parametrize("name", ["total_variation", "negativity", "std"])
def test_weak_slice_metrics_still_give_a_finite_curve(reconstruction_at, name):
    scan = scan_centers(name, TRUE + np.linspace(-6, 6, 9), reconstruct=lambda c: reconstruction_at(c))
    assert np.isfinite(scan.values).all() and scan.values.std() > 0


def test_a_curve_that_was_already_computed_can_be_shown_without_computing_it_again():
    from calibration import scan_from_curve

    centers = np.arange(20.0, 60.0)
    values = (centers - 41.3) ** 2 + 5.0
    scan = scan_from_curve("vo", centers, values)
    assert scan.best == pytest.approx(41.3, abs=0.05) and scan.quality[scan.best_index] == 1.0
    forced = scan_from_curve("vo", centers, values, best=41.25, approximate="note")
    assert forced.best == 41.25 and forced.approximate == "note" and forced.best_index == scan.best_index
