"""calibration.search_center(): a steerable (guess + tolerance) center search that returns scored candidates."""
import numpy as np
import pytest
from scipy.ndimage import rotate

from calibration import CenterSearch, search_center, search_window
from calibration.backend import CalibrationBackend


def test_search_window_rules():
    assert search_window(200) == (50.0, 150.0)                                  # default: middle half
    assert search_window(200, guess=120, tolerance=15) == (105.0, 135.0)
    assert search_window(200, guess=120) == (100.0, 140.0)                      # tolerance defaults to width/10
    assert search_window(200, guess=5, tolerance=30) == (0.0, 35.0)             # clipped to the detector
    assert search_window(200, guess=199, tolerance=30)[1] == 199.0
    assert search_window(200, tolerance=10) == (90.0, 110.0)                    # tolerance alone: around the middle
    assert search_window(200, guess=120, tolerance=15, start=10, stop=20) == (10.0, 20.0)  # explicit wins


class _FixedBackend(CalibrationBackend):
    name = "fixed"

    def find_center(self, sino_180, start=None, stop=None, **options):
        return 42.0


def test_a_backend_without_scoring_still_gives_one_candidate():
    result = search_center(np.zeros((10, 100)), _FixedBackend())
    assert isinstance(result, CenterSearch) and result.best == 42.0
    assert len(result.candidates) == 1 and np.isnan(result.candidates[0].score)


algotom = pytest.importorskip("algotom")


@pytest.fixture(scope="module")
def sinogram():
    """Attenuation sinogram of a few disks whose rotation axis is at column 147.5."""
    n, n_angles = 192, 180
    y, x = np.mgrid[:n, :n] - n / 2
    img = np.zeros((n, n), np.float32)
    for cx, cy, r, v in [(0, 0, .40, 1.0), (.1, .05, .15, -.5), (-.15, -.1, .10, .8), (.0, -.2, .06, 1.5)]:
        img[(x - cx * n) ** 2 + (y - cy * n) ** 2 < (r * n) ** 2] += v
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    line = np.stack([rotate(img, -a, reshape=False, order=1).sum(axis=0) for a in angles]).astype(np.float32)
    return np.roll(line, 20, axis=1) * (3.0 / line.max()), (n - 1) / 2.0 + 20


def test_automatic_search_finds_the_axis_and_lists_weaker_alternatives(sinogram):
    sino, true = sinogram
    result = search_center(sino, "algotom")
    assert result.best == pytest.approx(true, abs=0.5)
    assert result.candidates[0].score == 1.0
    assert all(c.score < 0.5 for c in result.candidates[1:])
    assert [c.score for c in result.candidates] == sorted((c.score for c in result.candidates), reverse=True)
    assert not result.at_edge


def test_the_search_hands_over_the_curve_it_evaluated(sinogram):
    sino, true = sinogram
    result = search_center(sino, "algotom")
    centers, metrics = result.curve
    assert len(centers) == len(metrics) and centers[0] == result.start and centers[-1] == result.stop       # every column
    assert np.all(np.diff(centers) == 1.0)
    assert centers[int(np.argmin(metrics))] == pytest.approx(true, abs=1.0)
    assert search_center(np.zeros((10, 100)), _FixedBackend()).curve is None       # a backend that keeps none


def test_a_guess_with_tolerance_restricts_the_search(sinogram):
    sino, true = sinogram
    result = search_center(sino, "algotom", guess=true - 8, tolerance=15)
    assert result.best == pytest.approx(true, abs=0.5)
    assert (result.start, result.stop) == (true - 23, true + 7)


def test_a_window_that_misses_the_axis_is_flagged_not_trusted(sinogram):
    sino, true = sinogram
    result = search_center(sino, "algotom", guess=true - 60, tolerance=10)
    assert result.at_edge  # "the minimum is at the edge of what you let me search"
    wider = search_center(sino, "algotom", guess=true - 20, tolerance=40)  # now the window contains the axis
    assert wider.best == pytest.approx(true, abs=0.5) and not wider.at_edge


def test_search_leaves_the_input_untouched(sinogram):
    sino, _ = sinogram
    before = sino.copy()
    search_center(sino, "algotom")
    np.testing.assert_array_equal(sino, before)
