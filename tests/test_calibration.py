"""
Tests for calibration/: the registry, the public find_center()/
find_tilt()/centers_for_rows() dispatch (row sampling, outlier
rejection, the line fit, and the decoupled correction arithmetic), and
a real end-to-end check of the algotom backend against the phantom
generator's axis-lean simulation (skipped if algotom isn't installed).

Dispatch tests use a hand-written sequence backend passed directly as
`backend=` (find_center()/find_tilt() both accept an instance, not just
a name), so they don't touch the global registry or require algotom at
all.
"""
import numpy as np
import pytest

from calibration import TiltResult, centers_for_rows, find_center, find_tilt, get_backend
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


def test_find_tilt_fits_a_line_through_per_row_estimates():
    # 5 rows, perfectly linear centers -- slope=2 px/row, intercept=10
    backend = _SequenceBackend([10.0, 12.0, 14.0, 16.0, 18.0])
    projections = np.zeros((20, 5, 30))  # (n_angles, n_rows, width) -- content unused by this fake backend

    result = find_tilt(projections, backend=backend, num_rows=5, margin=0)

    assert isinstance(result, TiltResult)
    assert result.rows.tolist() == [0, 1, 2, 3, 4]
    assert np.allclose(result.centers, [10, 12, 14, 16, 18])
    assert np.all(result.used)
    assert np.isclose(result.tilt_deg, np.degrees(np.arctan(2.0)))
    # slope*mid_row + intercept, mid_row=(5-1)/2=2
    assert np.isclose(result.center_at_reference_row, 2.0 * 2.0 + 10.0)
    assert result.reference_row == 2.0


def test_find_tilt_drops_outlier_rows():
    # rows 0,1,3,4 are linear (slope=2, intercept=10); row 2 is garbage
    backend = _SequenceBackend([10.0, 12.0, 999.0, 16.0, 18.0])
    projections = np.zeros((20, 5, 30))

    result = find_tilt(projections, backend=backend, num_rows=5, margin=0)

    assert result.used.tolist() == [True, True, False, True, True]
    assert np.isclose(result.tilt_deg, np.degrees(np.arctan(2.0)))


def test_find_tilt_raises_if_too_few_rows_survive():
    backend = _SequenceBackend([10.0, 500.0, 12.0, -500.0, 800.0])
    projections = np.zeros((20, 5, 30))

    with pytest.raises(RuntimeError, match="too inconsistent"):
        find_tilt(projections, backend=backend, num_rows=5, margin=0)


def test_find_tilt_center_re_centers_the_search_window_not_the_data():
    """
    The actual point of `center`: every row's find_center() call should
    search around `center`, not the sinogram's nominal width/2 -- and
    the projections array itself must be passed through unmodified (an
    earlier version shifted the array with scipy.ndimage.shift, which
    corrupted good rows at large offsets; this locks in that it no
    longer does that)."""
    backend = _SequenceBackend([10.0, 12.0, 14.0, 16.0, 18.0])
    projections = np.zeros((20, 5, 30))

    find_tilt(projections, backend=backend, center=25.0, num_rows=5, margin=0)

    for call in backend.calls:
        assert call["start"] == 25.0 - 30 / 4.0
        assert call["stop"] == 25.0 + 30 / 4.0
        assert call["shape"] == (20, 30)  # unmodified sinogram shape -- no shift applied to the data


def test_find_tilt_center_does_not_override_an_explicit_start_stop():
    backend = _SequenceBackend([10.0, 10.0, 10.0])
    projections = np.zeros((20, 3, 30))

    find_tilt(projections, backend=backend, center=25.0, start=1.0, stop=2.0, num_rows=3, margin=0)

    assert backend.calls[0]["start"] == 1.0
    assert backend.calls[0]["stop"] == 2.0


def test_find_tilt_rejects_too_small_a_margin():
    backend = _SequenceBackend([0.0])
    projections = np.zeros((20, 4, 30))

    with pytest.raises(ValueError, match="Not enough rows"):
        find_tilt(projections, backend=backend, num_rows=3, margin=2)


def test_centers_for_rows_is_flat_when_tilt_is_zero():
    centers = centers_for_rows(tilt_deg=0.0, center=50.0, reference_row=5.0, rows=[0, 5, 10])

    assert np.allclose(centers, [50.0, 50.0, 50.0])


def test_centers_for_rows_matches_a_known_slope():
    # slope=2 px/row, center=50 at reference_row=5
    centers = centers_for_rows(
        tilt_deg=np.degrees(np.arctan(2.0)), center=50.0, reference_row=5.0, rows=[0, 5, 10, 20],
    )

    # center(row) = 50 + 2*(row - 5)
    assert np.allclose(centers, [40.0, 50.0, 60.0, 80.0])


def test_centers_for_rows_does_not_need_reference_row_among_the_requested_rows():
    centers = centers_for_rows(
        tilt_deg=np.degrees(np.arctan(1.0)), center=100.0, reference_row=50.0, rows=[0, 200],
    )

    assert np.allclose(centers, [100.0 + (0 - 50.0), 100.0 + (200 - 50.0)])


def test_centers_for_rows_accepts_manually_determined_values():
    """
    The actual point of centers_for_rows() taking plain numbers, not a
    TiltResult: a tilt from find_tilt() and a center from anywhere else
    -- find_center() at a specific row, or a value someone determined
    by hand -- combine with no friction, because neither argument
    depends on how the other was obtained.
    """
    manual_center = 47.5  # e.g. eyeballed from a visual center check, not from find_center()
    manual_reference_row = 12.0  # the row that manual_center was determined at

    centers = centers_for_rows(
        tilt_deg=np.degrees(np.arctan(0.5)), center=manual_center, reference_row=manual_reference_row, rows=[12, 14],
    )

    assert np.allclose(centers, [47.5, 47.5 + 0.5 * 2])


algotom = pytest.importorskip("algotom", reason="algotom not installed -- the algotom calibration backend is optional")


def test_find_tilt_recovers_a_known_axis_lean():
    """
    Real end-to-end check: generate a phantom with a KNOWN rotation-axis
    tilt and center-of-rotation deviation, normalize it, and confirm
    find_tilt() recovers the tilt (and its own free center cross-check)
    to a tight tolerance -- this is the exact experiment that validated
    this method in the first place (see calibration/api.py's module
    docstring), now locked in as a regression test.
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
    result = find_tilt(normalized, backend="algotom", num_rows=14, margin=4, outlier_tolerance=6.0)

    assert result.tilt_deg == pytest.approx(true_tilt_deg, abs=1.5)
    assert result.center_at_reference_row == pytest.approx(true_center + 48.0, abs=1.0)  # width/2 + true_center


def test_find_tilt_center_recovers_a_disproportionate_offset_the_default_range_would_miss():
    """
    This is a deliberately UNREALISTIC edge case, not typical usage --
    see test_find_tilt_center_is_a_no_op_at_a_realistic_offset below for
    what actually happens in practice. 25px on a 96px-wide phantom is a
    ~26% relative offset; a real detector's worst-case offsets (tens of
    px on a 1200-2000px detector) are more like 1-4%, and at that
    realistic proportion `center` makes no measurable difference at all
    (confirmed directly, see the other test). This one exists only to
    lock in that the underlying failure mode -- a large enough offset
    pushing the highest-signal rows outside find_center()'s default
    window -- is real and that `center` genuinely fixes it, for the
    rare case where a real misalignment actually is this severe.
    """
    from preprocessing import build_reference, normalize_rows
    from tests.fixtures.realistic_phantom import generate, to_project_object

    true_tilt_deg = 1.0
    true_cor = 25.0
    phantom = generate(
        shape=(160, 96, 96), n_angles=300,
        cor_deviation=true_cor, tilt_deg=true_tilt_deg,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0,
        noise_level=0.0, seed=13,
        # Pinned explicitly: this test's numeric thresholds below were
        # validated at this exposure, not generate()'s auto-calibrated
        # default (which targets a realistic minimum transmission, see
        # doc/REALISTIC_PHANTOM.md, and is unrelated to what this test
        # is actually checking).
        attenuation_scale=2.0 * 64 / 96,
    )
    project = to_project_object(phantom)
    reference = build_reference(project)
    normalized = normalize_rows(project, slice(None), reference)
    middle_row = normalized.shape[1] // 2

    without_hint = find_tilt(normalized, backend="algotom", num_rows=40, margin=20, outlier_tolerance=6.0)

    coarse_center = find_center(normalized[:, middle_row, :], backend="algotom")
    with_hint = find_tilt(
        normalized, backend="algotom", center=coarse_center, num_rows=40, margin=20, outlier_tolerance=6.0,
    )

    assert with_hint.tilt_deg == pytest.approx(true_tilt_deg, abs=0.3)
    assert abs(with_hint.tilt_deg - true_tilt_deg) < abs(without_hint.tilt_deg - true_tilt_deg)


def test_find_tilt_center_is_a_no_op_at_a_realistic_offset():
    """
    What actually happens in the common case, not the edge case above:
    a proportionally realistic center-of-rotation offset (~3% of width
    here -- equivalent to ~45px on a real 1500px-wide detector) is
    already comfortably inside find_center()'s default search window,
    so `center` changes nothing. This is the honest, typical outcome --
    `center` exists for the rare severe-misalignment case, not because
    ordinary calibration runs need it.
    """
    from preprocessing import build_reference, normalize_rows
    from tests.fixtures.realistic_phantom import generate, to_project_object

    phantom = generate(
        shape=(160, 96, 96), n_angles=300,
        cor_deviation=3.0, tilt_deg=1.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0,
        noise_level=0.0, seed=13,
    )
    project = to_project_object(phantom)
    reference = build_reference(project)
    normalized = normalize_rows(project, slice(None), reference)
    middle_row = normalized.shape[1] // 2

    without_hint = find_tilt(normalized, backend="algotom", num_rows=40, margin=20, outlier_tolerance=6.0)
    coarse_center = find_center(normalized[:, middle_row, :], backend="algotom")
    with_hint = find_tilt(
        normalized, backend="algotom", center=coarse_center, num_rows=40, margin=20, outlier_tolerance=6.0,
    )

    assert with_hint.tilt_deg == without_hint.tilt_deg


def test_find_center_at_middle_row_agrees_with_find_tilt_s_cross_check():
    """
    The recommended workflow (center from find_center() at the middle
    row, tilt from find_tilt()) should agree with find_tilt()'s own
    free center cross-check at the same row -- if it didn't, one of the
    two methods would be wrong. Confirms the two are safe to mix, not
    just independently correct.
    """
    from preprocessing import build_reference, normalize_rows
    from tests.fixtures.realistic_phantom import generate, to_project_object

    phantom = generate(
        shape=(24, 96, 96), n_angles=300,
        cor_deviation=4.0, tilt_deg=10.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0,
        noise_level=0.0, seed=11,
    )
    project = to_project_object(phantom)
    reference = build_reference(project)
    normalized = normalize_rows(project, slice(None), reference)
    n_rows = normalized.shape[1]
    middle_row = (n_rows - 1) // 2

    tilt_result = find_tilt(normalized, backend="algotom", num_rows=14, margin=4, outlier_tolerance=6.0)
    center_at_middle = find_center(normalized[:, middle_row, :], backend="algotom")

    assert center_at_middle == pytest.approx(tilt_result.center_at_reference_row, abs=1.5)


def test_centers_for_rows_correction_improves_reconstruction_on_a_real_tilted_phantom():
    """
    The actual point of centers_for_rows(): reconstructing a tilted
    phantom's rows with a single constant center should correlate worse
    against ground truth than reconstructing with the per-row centers
    it computes -- checked on the row farthest from the reference row,
    where an uncorrected tilt does the most damage, the same experiment
    doc/demo/04_reconstruction.ipynb runs and visually confirms. Uses
    the recommended manual-friendly combo: find_center() at the middle
    row for the center, find_tilt() for the tilt.
    """
    from preprocessing import build_reference, normalize_rows
    from reconstruction import Geometry, reconstruct
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
    n_rows = normalized.shape[1]
    middle_row = (n_rows - 1) // 2

    tilt_result = find_tilt(normalized, backend="algotom", num_rows=14, margin=4, outlier_tolerance=6.0)
    center_at_middle = find_center(normalized[:, middle_row, :], backend="algotom")

    far_row = 19  # near the top of the sampled range -- far from the middle row
    sinogram = normalized[:, far_row, :]
    geometry = Geometry(geometry_type="parallel")

    single_center = reconstruct(sinogram, phantom.angles, center=center_at_middle, geometry=geometry, backend="algotom_fbp")
    per_row_center = float(
        centers_for_rows(tilt_result.tilt_deg, center_at_middle, reference_row=middle_row, rows=[far_row])[0]
    )
    corrected = reconstruct(sinogram, phantom.angles, center=per_row_center, geometry=geometry, backend="algotom_fbp")

    ground_truth = phantom.volume[far_row]
    corr_uncorrected = np.corrcoef(single_center.ravel(), ground_truth.ravel())[0, 1]
    corr_corrected = np.corrcoef(corrected.ravel(), ground_truth.ravel())[0, 1]

    assert corr_corrected > corr_uncorrected


def test_find_center_is_unaffected_by_reconstruction_backend_nondeterminism():
    """
    This module never reconstructs anything -- find_center() is pure
    sinogram-domain math. Calling it twice on identical input must give
    the identical result, unlike reconstruction.reconstruct() on the
    CPU-fallback FBP path (verified separately to vary run to run)."""
    from preprocessing import build_reference, normalize_rows
    from tests.fixtures.realistic_phantom import generate, to_project_object

    phantom = generate(
        shape=(8, 96, 96), n_angles=300, cor_deviation=5.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0,
        noise_level=0.0, seed=3,
    )
    project = to_project_object(phantom)
    reference = build_reference(project)
    normalized = normalize_rows(project, slice(None), reference)
    sino_180 = normalized[:, 4, :]

    a = find_center(sino_180, backend="algotom")
    b = find_center(sino_180, backend="algotom")
    assert a == b
