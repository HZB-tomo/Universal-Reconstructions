"""
Where does the CPU gridrec spend its time? Per-stage seconds for a run on a
Shepp-Logan stack (gridrec's own `timings` hook).

    python benchmarks/profile_cpu.py --width 1024 --slices 16
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from bench_phantom import sinogram  # noqa: E402
from urct_gridrec import gridrec, make_plan  # noqa: E402

STAGES = ("pad+log", "rfft", "spectrum", "scatter", "fold+shift", "ifft2+crop", "deapod+out")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--slices", type=int, default=16)
    ap.add_argument("--angles", type=int, default=0, help="default: width * pi / 2")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--oversampling", type=float, default=1.25)
    ap.add_argument("--kernel", type=int, default=6)
    args = ap.parse_args()

    width = args.width
    n_angles = args.angles or int(round(width * np.pi / 2))
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    line = sinogram(width, angles)
    stack = np.repeat(np.exp(-line * (3.0 / line.max()))[:, None, :], args.slices, axis=1).astype(np.float32)
    plan = make_plan(width, args.oversampling, args.kernel)
    center = (width - 1) / 2.0

    gridrec(stack, angles, center, plan=plan)                       # warm-up (JIT, FFT plans)
    best = None
    for _ in range(args.repeats):
        timings: dict = {}
        t = time.perf_counter()
        gridrec(stack, angles, center, plan=plan, timings=timings)
        total = time.perf_counter() - t
        if best is None or total < best[0]:
            best = (total, timings)
    total, timings = best
    print(f"width {width}, {n_angles} angles, {args.slices} slices; N={plan.n}, kernel {plan.kernel.width}, grid {plan.padded}^2")
    print(f"total {total:.3f} s  ({total / args.slices * 1000:.1f} ms/slice)  (best of {args.repeats})")
    for name in STAGES:
        print(f"  {name:12s} {timings.get(name, 0.0):7.3f} s  {100 * timings.get(name, 0.0) / total:5.1f}%")
    rest = total - sum(timings.get(n, 0.0) for n in STAGES)
    print(f"  {'unaccounted':12s} {rest:7.3f} s  {100 * rest / total:5.1f}%")


if __name__ == "__main__":
    main()
