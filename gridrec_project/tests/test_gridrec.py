"""urct_gridrec: gridding accuracy, absolute scale, geometry, reproducibility."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import rotate

from urct_gridrec import gridrec, make_plan
from urct_gridrec.filters import FILTERS, filter_response
from urct_gridrec.kernel import Kernel
from urct_gridrec.reference import direct_reconstruction

ROOT = Path(__file__).resolve().parents[1]


def phantom(width):
    y, x = np.mgrid[:width, :width] - (width - 1) / 2
    r = width / 32.0
    return (
        ((x - 3 * r) ** 2 + (y + 2 * r) ** 2 < (8 * r) ** 2) * 1.0
        + ((x + 7 * r) ** 2 + (y - 5 * r) ** 2 < (4 * r) ** 2) * 0.5
    ).astype(np.float32)


def sinogram(img, n_angles, shift=0):
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    line = np.stack([rotate(img, -a, reshape=False, order=1).sum(axis=0) for a in angles]).astype(np.float32)
    return np.roll(line, shift, axis=1), angles


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.fixture(scope="module")
def small():
    img = phantom(32)
    line, angles = sinogram(img, 48)
    return img, line, angles


def test_kernel_table_is_normalised_and_its_transform_is_smooth():
    k = Kernel.build(6, 2.0)
    assert k.table[k.table.size // 2] == pytest.approx(1.0)
    assert k.table[0] < 1e-3 and k.table[-1] < 1e-3
    ft = k.transform(np.linspace(0, 0.25, 6))
    assert np.all(np.diff(ft) < 0) and ft[0] > 0       # decreasing, positive over the image band


def test_closed_form_kernel_transform_matches_numerical_integration():
    for width, sigma in ((4, 1.25), (6, 1.25), (6, 2.0), (8, 1.5)):
        k = Kernel.build(width, sigma)
        s = np.linspace(0.0, 0.3, 25)
        assert np.allclose(k.transform(s), k.transform_numeric(s), rtol=1e-6)


def test_plans_are_cached_and_equal_settings_share_one():
    assert make_plan(48) is make_plan(48)
    assert make_plan(48) is not make_plan(48, oversampling=2.0)


def test_filters_have_the_ramp_shape():
    n = 64
    for name in FILTERS:
        f = filter_response(name, n)
        assert f.shape == (n // 2 + 1,) and f[0] == pytest.approx(2 / np.pi ** 2)   # Ram-Lak DC term
    ramp = filter_response("ramp", n)
    assert np.allclose(ramp[1:], np.arange(1, n // 2 + 1))
    assert filter_response("hann", n)[-1] == pytest.approx(0.0, abs=1e-9)   # the window kills Nyquist
    with pytest.raises(ValueError):
        filter_response("nope", n)


def test_gridding_matches_the_exact_fourier_sum(small):
    """The reference gridrec differs from the exact (no gridding) sum only by gridding error."""
    _, line, angles = small
    exact = direct_reconstruction(line, np.radians(angles), 15.5, make_plan(32))
    gridded = gridrec(line, angles, 15.5, device="reference", apply_log=False, ratio=None)
    assert rel(gridded, exact) < 1e-3


def test_wider_kernel_means_smaller_gridding_error(small):
    _, line, angles = small
    exact = direct_reconstruction(line, np.radians(angles), 15.5, make_plan(32))
    errors = [
        rel(gridrec(line, angles, 15.5, device="reference", apply_log=False, ratio=None, kernel_width=w), exact)
        for w in (2, 4, 8)
    ]
    assert errors[0] > errors[1] > errors[2]


def test_numba_cpu_equals_the_reference(small):
    _, line, angles = small
    stack = np.stack([line, 0.5 * line, line], axis=1)       # a pair plus a lone slice
    a = gridrec(stack, angles, 15.5, device="cpu", apply_log=False, ratio=None)
    b = gridrec(stack, angles, 15.5, device="reference", apply_log=False, ratio=None)
    assert a.shape == (3, 32, 32) and a.dtype == np.float32
    assert rel(a, b) < 1e-5


def test_absolute_scale_and_shape_of_the_reconstruction(small):
    img, line, angles = small
    out = gridrec(line, angles, 15.5, apply_log=False, ratio=None, filter_name="ramp")
    yy, xx = np.mgrid[:32, :32]
    inside = (xx - 15.5 - 3) ** 2 + (yy - 15.5 + 2) ** 2 < 5 ** 2
    assert out[inside].mean() == pytest.approx(1.0, abs=0.06)      # attenuation values come back
    assert np.corrcoef(out.ravel(), img.ravel())[0, 1] > 0.95
    assert abs(out[0, 0]) < 0.1                                   # background stays near zero


@pytest.mark.parametrize("width", [31, 32, 45])
def test_even_and_odd_widths(width):
    img = phantom(width)
    line, angles = sinogram(img, 60)
    out = gridrec(line, angles, (width - 1) / 2, apply_log=False, ratio=None)
    assert out.shape == (width, width)
    assert np.corrcoef(out.ravel(), img.ravel())[0, 1] > 0.93


def test_a_shifted_rotation_axis_is_handled_by_the_center_argument(small):
    _, line, angles = small
    straight = gridrec(line, angles, 15.5, apply_log=False, ratio=None)
    moved = gridrec(np.roll(line, 3, axis=1), angles, 18.5, apply_log=False, ratio=None)
    assert rel(moved, straight) < 0.02
    wrong = gridrec(np.roll(line, 3, axis=1), angles, 15.5, apply_log=False, ratio=None)
    assert rel(wrong, straight) > 3 * rel(moved, straight)


def test_per_slice_centers(small):
    _, line, angles = small
    stack = np.stack([line, np.roll(line, 2, axis=1)], axis=1)
    both = gridrec(stack, angles, np.array([15.5, 17.5]), apply_log=False, ratio=None)
    one = gridrec(stack[:, 1, :], angles, 17.5, apply_log=False, ratio=None)
    # Packing two slices into one complex transform leaks each into the other by about the
    # gridding error (6e-5 here at oversampling 1.25; 7e-7 at 2.0).
    assert rel(both[1], one) < 2e-4


def test_exactly_reproducible_and_input_untouched(small):
    _, line, angles = small
    stack = np.stack([line, line * 0.8], axis=1).copy()
    stack[3, 0, 5] = 0.0
    before = stack.copy()
    a = gridrec(stack, angles, 15.5, device="cpu")
    b = gridrec(stack, angles, 15.5, device="cpu")
    np.testing.assert_array_equal(a, b)           # race-free gridding: bit-for-bit repeatable
    np.testing.assert_array_equal(stack, before)
    # The GPU gridding adds into the grid with float32 atomics, whose order varies run to run:
    # repeatable to rounding only.
    c = gridrec(stack, angles, 15.5)
    d = gridrec(stack, angles, 15.5)
    np.testing.assert_allclose(c, d, atol=1e-5 * np.abs(c).max())
    np.testing.assert_array_equal(stack, before)


def test_circular_field_of_view_mask(small):
    _, line, angles = small
    out = gridrec(line, angles, 15.5, apply_log=False)
    assert out[0, 0] == 0.0 and out[-1, -1] == 0.0
    assert gridrec(line, angles, 15.5, apply_log=False, ratio=None)[0, 0] != 0.0


def test_bad_arguments():
    with pytest.raises(ValueError):
        gridrec(np.ones((4, 8)), np.arange(5.0), 3.5)
    with pytest.raises(ValueError):
        gridrec(np.ones((4, 8)), np.arange(4.0), 3.5, device="tpu")


_GPU_SIM = r"""
import os, sys, warnings
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
import numpy as np
from urct_gridrec import gridrec, gpu_available
assert gpu_available()
rng = np.random.default_rng(0)
stack = rng.uniform(0.2, 1.0, (20, 3, 24)).astype(np.float32)
angles = np.linspace(0, 180, 20, endpoint=False)
for centers in (11.5, np.array([11.0, 11.5, 12.0])):
    gpu = gridrec(stack, angles, centers, device="gpu")
    cpu = gridrec(stack, angles, centers, device="cpu")
    err = np.linalg.norm(gpu - cpu) / np.linalg.norm(cpu)
    assert gpu.shape == (3, 24, 24) and err < 1e-5, err
print("OK")
"""


def test_gpu_kernel_matches_cpu_on_the_cuda_simulator():
    result = subprocess.run(
        [sys.executable, "-c", _GPU_SIM, str(ROOT / "src")], capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stdout + result.stderr


def test_agrees_with_the_main_project_fbp_when_it_is_available():
    main = ROOT.parent
    if not (main / "reconstruction").is_dir():
        pytest.skip("main URCT project not next to this one")
    pytest.importorskip("algotom")
    sys.path.insert(0, str(main))
    from reconstruction import Geometry, reconstruct

    img = phantom(64)
    line, angles = sinogram(img, 120)
    trans = np.exp(-line * 0.05).astype(np.float32)
    stack = np.repeat(trans[:, None, :], 2, axis=1)
    fbp = reconstruct(
        stack, angles, 31.5, geometry=Geometry(geometry_type="parallel"), backend="algotom_fbp", device="cpu",
    )
    grid = gridrec(stack, angles, 31.5)
    assert np.corrcoef(fbp.ravel(), grid.ravel())[0, 1] > 0.995
    assert grid.mean() / fbp.mean() == pytest.approx(1.0, abs=0.03)    # same units, same amplitude
