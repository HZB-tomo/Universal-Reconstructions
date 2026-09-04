"""
Tests for reconstruction/: the registry, the public reconstruct()
dispatch (shape handling, geometry validation), and a real end-to-end
check of the algotom_fbp backend against the phantom generator (skipped
if algotom isn't installed).

Dispatch tests use a hand-written recording backend passed directly as
`backend=` (reconstruct() accepts an instance, not just a name), so they
don't touch the global registry or require algotom at all.
"""
import numpy as np
import pytest

from reconstruction import Geometry, get_backend, reconstruct
from reconstruction.backend import ReconstructionBackend


class _RecordingBackend(ReconstructionBackend):
    """Records exactly what it was called with; returns zeros of the
    right shape so shape-handling can be checked independently of any
    real reconstruction algorithm."""

    name = "recording"
    supported_geometries = {"parallel", "cone"}

    def __init__(self):
        self.calls = []

    def reconstruct(self, sinograms, angles, center, geometry, device=None, **options):
        self.calls.append(dict(shape=sinograms.shape, center=center, device=device, options=options))
        # sinograms is (n_angles, n_slices, n_columns) -- see backend.py
        n_slices, width = sinograms.shape[1], sinograms.shape[2]
        return np.zeros((n_slices, width, width), dtype=np.float32)


class _ConstantBackend(ReconstructionBackend):
    """Returns a known constant value everywhere, so unit conversion and
    dtype enforcement can be checked precisely -- unlike _RecordingBackend's
    zeros, which would hide a division bug behind 0/x == 0."""

    name = "constant"
    supported_geometries = {"parallel"}

    def __init__(self, value=2.0, dtype=np.float32):
        self.value = value
        self.dtype = dtype

    def reconstruct(self, sinograms, angles, center, geometry, device=None, **options):
        n_slices, width = sinograms.shape[1], sinograms.shape[2]
        return np.full((n_slices, width, width), self.value, dtype=self.dtype)


def _geometry(kind="parallel"):
    if kind == "parallel":
        return Geometry(geometry_type="parallel")
    return Geometry(geometry_type="cone", source_to_detector_distance=500.0, source_to_object_distance=100.0, detector_pixel_size=1.0)


def test_get_backend_resolves_the_builtin_algotom_backend():
    # Resolving by name works even without algotom installed -- only
    # actually calling .reconstruct() needs the real library.
    backend = get_backend("algotom_fbp")
    assert backend.name == "algotom_fbp"
    assert backend.supported_geometries == {"parallel"}


def test_get_backend_raises_for_unknown_name():
    with pytest.raises(KeyError, match="algotom_fbp"):
        get_backend("not_a_real_backend")


def test_reconstruct_single_sinogram_returns_2d_and_wraps_to_a_one_slice_stack():
    backend = _RecordingBackend()
    sinogram = np.random.default_rng(0).random((10, 20))  # (n_angles, width)
    angles = np.linspace(0, 180, 10, endpoint=False)

    image = reconstruct(sinogram, angles, center=10.0, geometry=_geometry(), backend=backend)

    assert image.ndim == 2
    assert image.shape == (20, 20)
    assert backend.calls[0]["shape"] == (10, 1, 20)  # (n_angles, n_slices=1, width)


def test_reconstruct_stack_returns_a_matching_3d_stack():
    backend = _RecordingBackend()
    stack = np.random.default_rng(0).random((10, 4, 20))  # (n_angles, n_slices, width) -- matches normalize_rows()'s own output shape
    angles = np.linspace(0, 180, 10, endpoint=False)

    images = reconstruct(stack, angles, center=10.0, geometry=_geometry(), backend=backend)

    assert images.shape == (4, 20, 20)  # (n_slices, height, width)
    assert backend.calls[0]["shape"] == (10, 4, 20)


def test_reconstruct_forwards_center_device_and_options_unchanged():
    backend = _RecordingBackend()
    sinogram = np.zeros((5, 8))
    angles = np.linspace(0, 180, 5, endpoint=False)

    reconstruct(sinogram, angles, center=3.5, geometry=_geometry(), backend=backend, device="gpu", filter_name="hann", num_iter=20)

    call = backend.calls[0]
    assert call["center"] == 3.5
    assert call["device"] == "gpu"
    assert call["options"] == {"filter_name": "hann", "num_iter": 20}


def test_reconstruct_accepts_per_slice_center_array():
    """The candidate-COR-search shape: many sinograms, a center per one --
    or, symmetrically here, just confirming the array passes through
    untouched for the backend to broadcast/use however it wants."""
    backend = _RecordingBackend()
    stack = np.zeros((5, 3, 8))  # (n_angles, n_slices, width)
    angles = np.linspace(0, 180, 5, endpoint=False)

    reconstruct(stack, angles, center=[3.0, 3.5, 4.0], geometry=_geometry(), backend=backend)

    assert backend.calls[0]["center"] == [3.0, 3.5, 4.0]


def test_reconstruct_returns_raw_backend_values_when_pixel_size_is_unset():
    backend = _ConstantBackend(value=2.0)
    sinogram = np.zeros((5, 8))
    angles = np.linspace(0, 180, 5, endpoint=False)
    geometry = Geometry(geometry_type="parallel")  # detector_pixel_size=None

    image = reconstruct(sinogram, angles, center=4.0, geometry=geometry, backend=backend)

    # "per pixel" units -- nothing to convert by, so left as-is.
    assert np.allclose(image, 2.0)


def test_reconstruct_divides_by_pixel_size_to_get_real_attenuation_units():
    backend = _ConstantBackend(value=2.0)
    sinogram = np.zeros((5, 8))
    angles = np.linspace(0, 180, 5, endpoint=False)
    geometry = Geometry(geometry_type="parallel", detector_pixel_size=4.0)  # um

    image = reconstruct(sinogram, angles, center=4.0, geometry=geometry, backend=backend)

    # 2.0 (1/px) / 4.0 (um/px) = 0.5 (1/um) -- see doc/RECONSTRUCTION.md's "Units" section
    assert np.allclose(image, 0.5)


def test_reconstruct_always_returns_float32_regardless_of_backend_dtype():
    backend = _ConstantBackend(value=2.0, dtype=np.float64)
    sinogram = np.zeros((5, 8))
    angles = np.linspace(0, 180, 5, endpoint=False)
    geometry = Geometry(geometry_type="parallel", detector_pixel_size=4.0)

    image = reconstruct(sinogram, angles, center=4.0, geometry=geometry, backend=backend)

    assert image.dtype == np.float32
    assert np.allclose(image, 0.5)


def test_reconstruct_rejects_unsupported_geometry():
    class _ParallelOnly(ReconstructionBackend):
        name = "parallel_only_test"
        supported_geometries = {"parallel"}

        def reconstruct(self, *args, **kwargs):
            raise AssertionError("should never be called -- geometry check must happen first")

    sinogram = np.zeros((5, 8))
    angles = np.linspace(0, 180, 5, endpoint=False)

    with pytest.raises(ValueError, match="parallel_only_test"):
        reconstruct(sinogram, angles, center=4.0, geometry=_geometry("cone"), backend=_ParallelOnly())


algotom = pytest.importorskip("algotom", reason="algotom not installed -- algotom_fbp backend is optional")


def test_algotom_fbp_reconstructs_a_recognizable_slice():
    """
    Real end-to-end check: generate a clean phantom (no injected COR/tilt
    error, no detector defects, so this test is purely about whether
    reconstruction itself works), normalize it, reconstruct one central
    slice, and confirm the result actually correlates with the known
    ground-truth cross-section -- not just that it runs without error.
    """
    from preprocessing import build_reference, normalize_rows
    from tests.fixtures.realistic_phantom import generate, to_project_object

    phantom = generate(
        shape=(8, 64, 64), n_angles=240,
        cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0,
        noise_level=0.0, seed=1,
    )
    project = to_project_object(phantom)
    reference = build_reference(project)
    normalized = normalize_rows(project, slice(None), reference)

    row = phantom.volume.shape[0] // 2
    sinogram = normalized[:, row, :]
    width = sinogram.shape[1]

    image = reconstruct(
        sinogram, phantom.angles, center=width / 2, geometry=_geometry(), backend="algotom_fbp",
    )

    assert image.shape == (width, width)
    correlation = np.corrcoef(image.ravel(), phantom.volume[row].ravel())[0, 1]
    # 0.7, not something closer to 1.0, deliberately -- verified directly (see
    # tests/fixtures/realistic_phantom/simulate.py's project_parallel docstring)
    # that FBP reconstruction of this specific phantom has a real, expected
    # correlation ceiling around 0.75-0.90 even in algotom's own best case
    # (Shepp-Logan's sharp discontinuities produce genuine Gibbs-ringing under
    # filtered backprojection at moderate resolution) -- this is a floor
    # confirming the pipeline is actually working, not a target for "perfect".
    assert correlation > 0.7


def test_normalize_rows_output_feeds_reconstruct_with_no_reshaping():
    """
    The actual point of matching normalize_rows()'s axis convention:
    its output -- (n_angles, n_slices, width) -- should be usable as
    reconstruct()'s input completely unmodified, no moveaxis/transpose
    needed at the call site.
    """
    from preprocessing import build_reference, normalize_rows
    from tests.fixtures.realistic_phantom import generate, to_project_object

    phantom = generate(
        shape=(6, 32, 32), n_angles=90,
        cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0,
        noise_level=0.0, seed=3,
    )
    project = to_project_object(phantom)
    reference = build_reference(project)

    rows = slice(1, 4)
    normalized = normalize_rows(project, rows, reference)  # (n_angles, 3, width) -- straight from preprocessing
    assert normalized.shape == (90, 3, 32)

    images = reconstruct(
        normalized, phantom.angles, center=16.0, geometry=_geometry(), backend="algotom_fbp",
    )

    assert images.shape == (3, 32, 32)  # (n_slices, height, width)
