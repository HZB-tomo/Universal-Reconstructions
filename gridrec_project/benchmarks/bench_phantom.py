"""
Gridrec vs filtered backprojection on a Shepp-Logan phantom with exact
(analytic) projections: accuracy against the known image, and speed.

    python benchmarks/bench_phantom.py                 # CPU
    python benchmarks/bench_phantom.py --device gpu    # on a machine with CuPy + a GPU
    python benchmarks/bench_phantom.py --sizes 256 512 --slices 8

FBP here is the main URCT project's `algotom_fbp` backend (found next to this
folder); skipped if it is not available.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "src"))
sys.path.insert(0, str(HERE.parents[1]))      # the main project, for the FBP backend
warnings.filterwarnings("ignore")

from urct_gridrec import gridrec, make_plan  # noqa: E402

# (x0, y0, a, b, phi_deg, mu) in units of the image half-width: modified Shepp-Logan
ELLIPSES = [
    (0.0, 0.0, 0.69, 0.92, 0, 1.0), (0.0, -0.0184, 0.6624, 0.874, 0, -0.8),
    (0.22, 0.0, 0.11, 0.31, -18, -0.2), (-0.22, 0.0, 0.16, 0.41, 18, -0.2),
    (0.0, 0.35, 0.21, 0.25, 0, 0.1), (0.0, 0.1, 0.046, 0.046, 0, 0.1),
    (0.0, -0.1, 0.046, 0.046, 0, 0.1), (-0.08, -0.605, 0.046, 0.023, 0, 0.1),
    (0.0, -0.606, 0.023, 0.023, 0, 0.1), (0.06, -0.605, 0.023, 0.046, 0, 0.1),
]


def phantom(width: int) -> np.ndarray:
    c = (width - 1) / 2.0
    ss = 3                                             # supersampling for the reference image
    ax = (np.arange(width * ss) + 0.5) / ss - 0.5 - c
    y, x = np.meshgrid(ax / (width / 2.0), ax / (width / 2.0), indexing="ij")
    img = np.zeros_like(x)
    for x0, y0, a, b, phi, mu in ELLIPSES:
        p = np.radians(phi)
        xr = (x - x0) * np.cos(p) + (y - y0) * np.sin(p)
        yr = -(x - x0) * np.sin(p) + (y - y0) * np.cos(p)
        img += mu * ((xr / a) ** 2 + (yr / b) ** 2 <= 1.0)
    return img.reshape(width, ss, width, ss).mean(axis=(1, 3)).astype(np.float32)


def sinogram(width: int, angles_deg: np.ndarray) -> np.ndarray:
    """Exact line integrals, in pixel units, in the detector convention of
    the reconstructions: detector position = x cos(t) - y sin(t) + center."""
    c = (width - 1) / 2.0
    half = width / 2.0
    s = np.arange(width) - c
    out = np.zeros((len(angles_deg), width))
    for x0, y0, a, b, phi, mu in ELLIPSES:
        x0, y0, a, b = x0 * half, y0 * half, a * half, b * half
        for j, t in enumerate(np.radians(angles_deg)):
            psi = -t                                   # ray direction in the (x, y) frame
            a2 = (a * np.cos(psi - np.radians(phi))) ** 2 + (b * np.sin(psi - np.radians(phi))) ** 2
            d = s - (x0 * np.cos(psi) + y0 * np.sin(psi))
            inside = a2 - d * d
            out[j] += mu * 2.0 * a * b / a2 * np.sqrt(np.clip(inside, 0.0, None))
    return out.astype(np.float32)


def timed(fn, repeats: int) -> float:
    fn()                                               # warm-up: JIT compilation, FFT plans, CUDA context
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--slices", type=int, default=16)
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu", "auto"])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    try:
        from reconstruction import Geometry, reconstruct
        fbp_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"(main project FBP not available: {e})")
        fbp_ok = False

    import numba
    print(f"gridrec device: {args.device}   numba threads: {numba.get_num_threads()}   slices per run: {args.slices}")
    header = f"{'width':>6} {'angles':>6} | {'gridrec s/slice':>16} {'FBP s/slice':>12} {'speedup':>8} | {'gridrec err':>11} {'FBP err':>8} | {'gridrec vs FBP':>14}"
    print(header)
    print("-" * len(header))
    for width in args.sizes:
        n_angles = int(round(width * np.pi / 2))
        angles = np.linspace(0, 180, n_angles, endpoint=False)
        truth = phantom(width)
        line = sinogram(width, angles)
        scale = 3.0 / line.max()                       # largest line integral = 3 (transmission down to ~5%)
        stack = np.repeat(np.exp(-line * scale)[:, None, :], args.slices, axis=1).astype(np.float32)
        truth_scaled = truth * scale
        center = (width - 1) / 2.0
        plan = make_plan(width)

        run_g = lambda: gridrec(stack, angles, center, device=args.device, plan=plan)
        t_g = timed(run_g, args.repeats) / args.slices
        out_g = run_g()[0]

        yy, xx = np.mgrid[:width, :width]
        inside = (xx - center) ** 2 + (yy - center) ** 2 <= (0.95 * width / 2) ** 2
        err = lambda im: float(np.sqrt(np.mean((im[inside] - truth_scaled[inside]) ** 2)) / np.sqrt(np.mean(truth_scaled[inside] ** 2)))

        if fbp_ok:
            geometry = Geometry(geometry_type="parallel")
            run_f = lambda: reconstruct(stack, angles, center, geometry=geometry, backend="algotom_fbp", device="cpu")
            t_f = timed(run_f, args.repeats) / args.slices
            out_f = run_f()[0]
            agree = float(np.corrcoef(out_g.ravel(), out_f.ravel())[0, 1])
            print(f"{width:>6} {n_angles:>6} | {t_g:>16.4f} {t_f:>12.4f} {t_f / t_g:>7.1f}x | {err(out_g):>10.1%} {err(out_f):>8.1%} | {agree:>14.5f}")
        else:
            print(f"{width:>6} {n_angles:>6} | {t_g:>16.4f} {'-':>12} {'-':>8} | {err(out_g):>10.1%} {'-':>8} | {'-':>14}")


if __name__ == "__main__":
    main()
