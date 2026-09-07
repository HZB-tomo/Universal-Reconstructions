"""
Tests for calibration/: the registry, the public find_center()/
find_tilt_and_center() dispatch (row sampling, outlier rejection, the
line fit), and a real end-to-end check of the algotom backend against
the phantom generator's axis-lean simulation (skipped if algotom isn't
installed).

Dispatch tests use a hand-written sequence backend passed directly as
`backend=` (both functions accept an instance, not just a name), so
they don't touch the global registry or require algotom at all.
"""
import numpy as np
import pytest

from calibration import TiltAndCenterResult, find_center, find_tilt_and_center, get_backend
from calibration.backend import CalibrationBackend


class _SequenceBackend(CalibrationBackend):
    """Returns pre-programmed values in call order -- lets a test fully
    control what "per-row centers" look like without needing real
    sinogram content or a real algorithm."""

    name = "sequence"

    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def find_center(self, sino_180, start=None, stop=None, **options):
        self.calls.append(dict(shape=sino_180.shape, start=start, stop=stop, options=options))
        return self.values[len(self.calls) - 1]


def test_get_backend_resolves_the_builtin_algotom_backend():
    # Resolving by name works even without algotom installed -- only
    # actually calling .find_center() needs the real library.
    backend = get_backend("algotom")
    assert backend.name == "algotom"


def test_get_backend_raises_for_unknown_name():
    with pytest.raises(KeyError, match="algotom"):
        get_backend("not_a_real_backend")


def test_find_center_defaults_the_search_range_to_a_quarter_width_around_center():
    backend = _SequenceBackend([42.0])
    sino_180 = np.zeros((10, 40))

    find_center(sino_180, backend=backend)

    assert backend.calls[0]["start"] == 10.0  # 40/2 - 40/4
    assert backend.calls[0]["stop"] == 30.0  # 40/2 + 40/4


def test_find_center_forwards_explicit_range_and_options_unchanged():
    backend = _SequenceBackend([42.0])
    sino_180 = np.zeros((10, 30))

    find_center(sino_180, backend=backend, start=5.0, stop=25.0, ratio=0.5)

    call = backend.calls[0]
    assert call["start"] == 5.0
    assert call["stop"] == 25.0
    assert call["options"] == {"ratio": 0.5}


def test_find_tilt_and_center_fits_a_line_through_per_row_estimates():
    # 5 rows, perfectly linear centers -- slope=2 px/row, intercept=10
    backend = _SequenceBackend([10.0, 12.0, 14.0, 16.0, 18.0])
    projections = np.zeros((20, 5, 30))  # (n_angles, n_rows, width) -- content unused by this fake backend

    result = find_tilt_and_center(projections, backend=backend, num_rows=5, margin=0)

    assert isinstance(result, TiltAndCenterResult)
    assert result.rows.tolist() == [0, 1, 2, 3, 4]
    assert np.allclose(result.centers, [10, 12, 14, 16, 18])
    assert np.all(result.used)
    assert np.isclose(result.tilt_deg, np.degrees(np.arctan(2.0)))
    assert np.isclose(result.center, 2.0 * 2.0 + 10.0)  # slope*mid_row + intercept, mid_row=(5-1)/2=2


def test_find_tilt_and_center_drops_outlier_rows():
    # rows 0,1,3,4 are linear (slope=2, intercept=10); row 2 is garbage
    backend = _SequenceBackend([10.0, 12.0, 999.0, 16.0, 18.0])
    projections = np.zeros((20, 5, 30))

    result = find_tilt_and_center(projections, backend=backend, num_rows=5, margin=0)

    assert result.used.tolist() == [True, True, False, True, True]
    assert np.isclose(result.tilt_deg, np.degrees(np.arctan(2.0)))


def test_find_tilt_and_center_raises_if_too_few_rows_survive():
    backend = _SequenceBackend([10.0, 500.0, 12.0, -500.0, 800.0])
    projections = np.zeros((20, 5, 30))

    with pytest.raises(RuntimeError, match="too inconsistent"):
        find_tilt_and_center(projections, backend=backend, num_rows=5, margin=0)


def test_find_tilt_and_center_rejects_too_small_a_margin():
    backend = _SequenceBackend([0.0])
    projections = np.zeros((20, 4, 30))

    with pytest.raises(ValueError, match="Not enough rows"):
        find_tilt_and_center(projections, backend=backend, num_rows=3, margin=2)


algotom = pytest.importorskip("algotom", reason="algotom not installed -- the algotom calibration backend is optional")


def test_find_tilt_and_center_recovers_a_known_axis_lean():
    """
    Real end-to-end check: generate a phantom with a KNOWN rotation-axis
    tilt and center-of-rotation deviation, normalize it, and confirm
    find_tilt_and_center() recovers both to a tight tolerance -- this is
    the exact experiment that validated this method in the first place
    (see calibration/api.py's module docstring), now locked in as a
    regression test.
    """
    from preprocessing import build_reference, normalize_rows
    from tests.fixtures.realistic_phantom import generate, to_project_object

    true_tilt_deg = 10.0
    true_center = 4.0
    phantom = generate(
        shape=(24, 96, 96), n_angles=300,
        cor_deviation=true_center, tilt_deg=true_tilt_deg,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0,
        noise_level=0.0, seed=11,
    )
    project = to_project_object(phantom)
    reference = build_reference(project)
    normalized = normalize_rows(project, slice(None), reference)

    # outlier_tolerance tightened from find_center()'s already-appropriately-wide
    # default range: this phantom's true deviations are only a few pixels, so a
    # much smaller tolerance than a real large-detector setup would use (see
    # Reconstruction_algotom_loop.ipynb's own 20px default) is what actually
    # catches a bad row here -- confirmed directly: row 12 of this phantom is a
    # genuine fluke (a local-minimum failure unrelated to the search range) that
    # a 20px tolerance would NOT catch at this small scale, but 6px does.
    result = find_tilt_and_center(normalized, backend="algotom", num_rows=14, margin=4, outlier_tolerance=6.0)

    assert result.tilt_deg == pytest.approx(true_tilt_deg, abs=1.5)
    assert result.center == pytest.approx(true_center + 48.0, abs=1.0)  # width/2 + true_center


def test_find_tilt_and_center_is_unaffected_by_reconstruction_backend_nondeterminism():
    """
    This module never reconstructs anything -- find_center() is pure
    sinogram-domain math. Calling it twice on identical input must give
    the identical result, unlike reconstruction.reconstruct() on the
    CPU-fallback FBP path (verified separately to vary run to run)."""
    from tests.fixtures.realistic_phantom import generate

    phantom = generate(
        shape=(8, 96, 96), n_angles=300, cor_deviation=5.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0,
        noise_level=0.0, seed=3,
    )
    from preprocessing import build_reference, normalize_rows
    from tests.fixtures.realistic_phantom import to_project_object

    project = to_project_object(phantom)
    reference = build_reference(project)
    normalized = normalize_rows(project, slice(None), reference)
    sino_180 = normalized[:, 4, :]

    a = find_center(sino_180, backend="algotom")
    b = find_center(sino_180, backend="algotom")
    assert a == b
