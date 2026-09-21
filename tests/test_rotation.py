"""
Tests for reconstruction/rotation.py: rotating a reconstructed volume
about the scan axis by relabeling the sinogram's angle axis
(shift_sinogram()/reconstruct_rotated()), with a fallback to rotating
the reconstructed volume directly when the scan's angular coverage
doesn't support an exact reorder (supports_exact_rotation() False).

Dispatch tests use a hand-written recording backend passed directly as
`backend=`, so they don't need algotom installed. The end-to-end tests
do (real reconstruction), and are skipped if it isn't.
"""
import numpy as np
import pytest
from scipy.ndimage import rotate as scipy_rotate

from reconstruction import Geometry, reconstruct, reconstruct_rotated
from reconstruction.backend import ReconstructionBackend
from reconstruction.rotation import shift_sinogram, supports_exact_rotation


# -- supports_exact_rotation() -------------------------------------------------

def test_supports_exact_rotation_parallel_180deg_coverage():
    angles = np.linspace(0, 180, 200, endpoint=False)
    assert supports_exact_rotation(angles, "parallel") is True


def test_supports_exact_rotation_parallel_360deg_coverage():
    angles = np.linspace(0, 360, 360, endpoint=False)
    assert supports_exact_rotation(angles, "parallel") is True


def test_supports_exact_rotation_parallel_limited_angle_is_unsupported():
    angles = np.linspace(0, 170, 250, endpoint=False)  # the 170deg case
    assert supports_exact_rotation(angles, "parallel") is False


def test_supports_exact_rotation_cone_360deg_coverage():
    angles = np.linspace(0, 360, 300, endpoint=False)
    assert supports_exact_rotation(angles, "cone") is True


def test_supports_exact_rotation_cone_180deg_coverage_is_unsupported():
    # unlike parallel-beam, there's no p(t+180,s)=p(t,-s) identity for a
    # divergent beam, so cone-beam needs the full 360deg period.
    angles = np.linspace(0, 180, 200, endpoint=False)
    assert supports_exact_rotation(angles, "cone") is False


def test_supports_exact_rotation_nonuniform_angles_is_unsupported():
    angles = np.array([0.0, 10.0, 10.5, 40.0, 90.0, 179.0])
    assert supports_exact_rotation(angles, "parallel") is False


def test_supports_exact_rotation_too_few_angles_is_unsupported():
    assert supports_exact_rotation(np.array([0.0]), "parallel") is False


# -- shift_sinogram() -----------------------------------------------------------

def test_shift_sinogram_returns_none_for_limited_angle():
    angles = np.linspace(0, 170, 20, endpoint=False)
    stack = np.zeros((20, 1, 4), dtype=np.float32)
    assert shift_sinogram(stack, angles, "parallel", shift_deg=5.0) is None


def test_shift_sinogram_grid_aligned_180deg_uses_the_flip_identity_at_the_wrap():
    # angles = 0, 45, 90, 135 -- span 135 = 180 - step, exactly the
    # standard "N angles over [0,180)" construction.
    angles = np.array([0.0, 45.0, 90.0, 135.0])
    stack = np.array(
        [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]], dtype=np.float32,
    )  # (n_angles=4, n_slices=1, width=3)

    result = shift_sinogram(stack, angles, "parallel", shift_deg=45.0)  # one whole step

    assert result is not None
    # new(a) = old(a - 1step): direct copy for a=1..3, no wrap.
    np.testing.assert_allclose(result[1, 0], [1, 2, 3])
    np.testing.assert_allclose(result[2, 0], [4, 5, 6])
    np.testing.assert_allclose(result[3, 0], [7, 8, 9])
    # a=0 wraps past 0deg: old(-45deg) = old(-45+180, flip s) = old(135) mirrored.
    np.testing.assert_allclose(result[0, 0], [12, 11, 10])


def test_shift_sinogram_360deg_span_is_a_plain_roll_no_mirror():
    angles = np.array([0.0, 90.0, 180.0, 270.0])
    stack = np.array(
        [[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]], dtype=np.float32,
    )

    result = shift_sinogram(stack, angles, "parallel", shift_deg=90.0)

    np.testing.assert_allclose(result[0, 0], [7, 8])  # old(270) = old(-90 mod 360)
    np.testing.assert_allclose(result[1, 0], [1, 2])
    np.testing.assert_allclose(result[2, 0], [3, 4])
    np.testing.assert_allclose(result[3, 0], [5, 6])


def test_shift_sinogram_off_grid_linearly_interpolates():
    angles = np.array([0.0, 45.0, 90.0, 135.0])
    stack = np.array(
        [[[0.0]], [[4.0]], [[8.0]], [[12.0]]], dtype=np.float32,
    )

    result = shift_sinogram(stack, angles, "parallel", shift_deg=22.5)  # half a step

    # new(2) = old(90 - 22.5) = old(67.5), halfway between old(45)=4 and old(90)=8.
    np.testing.assert_allclose(result[2, 0], [6.0], atol=1e-5)


def test_shift_sinogram_cone_360deg_matches_parallel_360deg_math():
    angles = np.array([0.0, 90.0, 180.0, 270.0])
    stack = np.array(
        [[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]], dtype=np.float32,
    )
    result = shift_sinogram(stack, angles, "cone", shift_deg=90.0)
    np.testing.assert_allclose(result[0, 0], [7, 8])


def test_shift_sinogram_cone_180deg_is_unsupported():
    angles = np.linspace(0, 180, 60, endpoint=False)
    stack = np.zeros((60, 1, 4), dtype=np.float32)
    assert shift_sinogram(stack, angles, "cone", shift_deg=10.0) is None


# -- reconstruct_rotated() dispatch (fake backends, no algotom needed) --------

class _RecordingBackend(ReconstructionBackend):
    """Records the exact sinogram data it was called with."""

    name = "rotation_test_recording"
    supported_geometries = {"parallel", "cone"}

    def __init__(self):
        self.calls = []

    def reconstruct(self, sinograms, angles, center, geometry, device=None, **options):
        self.calls.append(sinograms.copy())
        n_slices, width = sinograms.shape[1], sinograms.shape[2]
        return np.zeros((n_slices, width, width), dtype=np.float32)


class _FixedPatternBackend(ReconstructionBackend):
    """Ignores its input, always returns the same recognizable, asymmetric
    per-slice image -- so the fallback path's post-hoc rotation can be
    checked against an independently-computed scipy.ndimage.rotate call."""

    name = "rotation_test_fixed_pattern"
    supported_geometries = {"parallel", "cone"}

    def __init__(self, image):
        self.image = image

    def reconstruct(self, sinograms, angles, center, geometry, device=None, **options):
        n_slices = sinograms.shape[1]
        return np.stack([self.image] * n_slices)


def _geometry(kind="parallel"):
    if kind == "parallel":
        return Geometry(geometry_type="parallel")
    return Geometry(geometry_type="cone", source_to_detector_distance=500.0, source_to_object_distance=100.0, detector_pixel_size=None)


def test_reconstruct_rotated_uses_shifted_sinogram_when_coverage_supports_it():
    angles = np.linspace(0, 180, 40, endpoint=False)
    rng = np.random.default_rng(0)
    stack = rng.random((40, 2, 8)).astype(np.float32)
    backend = _RecordingBackend()
    expected = shift_sinogram(stack, angles, "parallel", shift_deg=9.0)  # one step (180/40=4.5 * 2)

    reconstruct_rotated(stack, angles, center=4.0, geometry=_geometry(), backend=backend, rotate_deg=9.0)

    assert len(backend.calls) == 1
    np.testing.assert_allclose(backend.calls[0], expected)


def test_reconstruct_rotated_falls_back_to_image_rotation_when_coverage_is_insufficient():
    angles = np.linspace(0, 170, 60, endpoint=False)  # limited angle
    rng = np.random.default_rng(1)
    stack = rng.random((60, 3, 16)).astype(np.float32)
    image = rng.random((16, 16)).astype(np.float32)
    backend = _FixedPatternBackend(image)

    result = reconstruct_rotated(stack, angles, center=8.0, geometry=_geometry(), backend=backend, rotate_deg=17.0)

    expected_slice = scipy_rotate(image, angle=17.0, reshape=False, order=1, mode="constant")
    for z in range(3):
        np.testing.assert_allclose(result[z], expected_slice)


def test_reconstruct_rotated_cone_beam_360deg_uses_shift():
    angles = np.linspace(0, 360, 100, endpoint=False)
    rng = np.random.default_rng(2)
    stack = rng.random((100, 1, 8)).astype(np.float32)
    backend = _RecordingBackend()
    expected = shift_sinogram(stack, angles, "cone", shift_deg=25.0)

    reconstruct_rotated(stack, angles, center=4.0, geometry=_geometry("cone"), backend=backend, rotate_deg=25.0)

    np.testing.assert_allclose(backend.calls[0], expected)


def test_reconstruct_rotated_cone_beam_180deg_falls_back():
    angles = np.linspace(0, 180, 100, endpoint=False)  # short of cone-beam's 360deg requirement
    rng = np.random.default_rng(3)
    stack = rng.random((100, 1, 8)).astype(np.float32)
    image = rng.random((8, 8)).astype(np.float32)
    backend = _FixedPatternBackend(image)

    result = reconstruct_rotated(stack, angles, center=4.0, geometry=_geometry("cone"), backend=backend, rotate_deg=25.0)

    expected = scipy_rotate(image, angle=25.0, reshape=False, order=1, mode="constant")
    np.testing.assert_allclose(result[0], expected)


def test_reconstruct_rotated_single_sinogram_matches_stack_form():
    angles = np.linspace(0, 180, 40, endpoint=False)
    rng = np.random.default_rng(4)
    sinogram = rng.random((40, 8)).astype(np.float32)  # 2D: one sinogram, no slice axis
    backend = _RecordingBackend()

    result = reconstruct_rotated(sinogram, angles, center=4.0, geometry=_geometry(), backend=backend, rotate_deg=4.5)

    assert result.ndim == 2
    assert backend.calls[0].shape == (40, 1, 8)


# -- end-to-end, real backend --------------------------------------------------

algotom = pytest.importorskip("algotom", reason="algotom not installed -- these tests need the algotom_fbp backend")

from preprocessing import build_reference, normalize_rows  # noqa: E402
from tests.fixtures.realistic_phantom import generate, to_project_object  # noqa: E402


def _normalized_stack(**generate_kwargs):
    phantom = generate(**generate_kwargs)
    project = to_project_object(phantom)
    reference = build_reference(project)
    normalized = normalize_rows(project, slice(None), reference)
    return phantom, normalized


def _corr(a, b) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def test_reconstruct_rotated_matches_image_domain_rotation_for_a_180deg_scan():
    """
    supports_exact_rotation() is True here (180deg coverage), so
    reconstruct_rotated() takes the sinogram-reorder path. It should
    still agree with independently reconstructing plainly and rotating
    the volume with scipy.ndimage.rotate -- the same self-calibrated
    "at least as good as algotom's own repeat-call noise floor" check
    test_streaming.py uses, since algotom_fbp isn't bit-deterministic
    call to call.
    """
    phantom, normalized = _normalized_stack(
        shape=(8, 64, 64), n_angles=150, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=7,
    )
    width = normalized.shape[-1]
    center = width / 2.0
    geometry = Geometry(geometry_type="parallel")
    theta = 24.0  # not required to be grid-aligned

    vol_base = reconstruct(normalized, phantom.angles, center, geometry, "algotom_fbp", device="cpu")
    vol_base_repeat = reconstruct(normalized, phantom.angles, center, geometry, "algotom_fbp", device="cpu")
    vol_rotated = reconstruct_rotated(
        normalized, phantom.angles, center, geometry, "algotom_fbp", rotate_deg=theta, device="cpu",
    )

    vol_image_rotated = scipy_rotate(vol_base, angle=theta, axes=(1, 2), reshape=False, order=1, mode="constant")

    baseline_noise_floor = _corr(vol_base, vol_base_repeat)
    cross = _corr(vol_image_rotated, vol_rotated)
    assert cross >= baseline_noise_floor - 0.05, (
        f"reconstruct_rotated() ({cross:.4f}) disagrees with an independent image-domain rotation "
        f"by more than algotom's own repeat-call noise floor ({baseline_noise_floor:.4f}) allows"
    )


def test_reconstruct_rotated_limited_angle_scan_falls_back_and_still_runs():
    """No p(t+180,s)=p(t,-s) redundancy exists for a genuine limited-angle
    (170deg) scan, so supports_exact_rotation() must be False here and
    reconstruct_rotated() must use the image-domain fallback -- this
    just confirms that actually happens end-to-end, on real
    (algotom_fbp) reconstructed data, without erroring."""
    phantom, normalized = _normalized_stack(
        shape=(4, 48, 48), n_angles=80, angle_span=170.0, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=8,
    )
    width = normalized.shape[-1]
    center = width / 2.0
    geometry = Geometry(geometry_type="parallel")

    assert supports_exact_rotation(phantom.angles, "parallel") is False

    volume = reconstruct_rotated(
        normalized, phantom.angles, center, geometry, "algotom_fbp", rotate_deg=15.0, device="cpu",
    )
    assert volume.shape == (4, width, width)
    assert np.isfinite(volume).all()
