"""
Where does FBP time go? Run on the machine you are asking about:

    python benchmarks/bench_fbp.py --width 2048 --angles 1800 --slices 69
    python benchmarks/bench_fbp.py --width 2048 --angles 1800 --slices 69 --device cpu

Prints, for a synthetic stack of the given size,
  - the ramp-filter time (always CPU, per stack),
  - the full algotom_fbp backend time on the chosen device,
  - throughput in slices/s and in billions of backprojection updates per
    second (width^2 x angles x slices / time) -- the number to compare
    between machines, since it does not depend on the data size.
The first call is a warm-up (JIT compilation / CUDA context) and not timed.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from reconstruction import Geometry, reconstruct, resolve_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--angles", type=int, default=1000)
    parser.add_argument("--slices", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    device = None if args.device == "auto" else args.device

    rng = np.random.default_rng(0)
    # Smooth positive "transmission" data: the values do not change the timing.
    stack = (0.5 + 0.4 * rng.random((args.angles, args.slices, args.width))).astype(np.float32)
    angles = np.linspace(0, 180, args.angles, endpoint=False)
    geometry = Geometry(geometry_type="parallel")
    center = (args.width - 1) / 2.0
    used = resolve_device("algotom_fbp", device)
    updates = float(args.width) ** 2 * args.angles * args.slices

    print(f"stack: {args.angles} angles x {args.slices} slices x {args.width} wide   device used: {used.upper()}")

    from algotom.rec import reconstruction as rec

    # -- ramp filter alone (what the CPU does before the backprojection)
    pad = min(int(0.15 * args.width), 150)
    ramp = rec.make_2d_ramp_window(args.angles, args.width + 2 * pad, "hann")
    rec.apply_ramp_filter(-np.log(stack[:, 0, :]), ramp, "hann", pad, "edge")
    t = time.perf_counter()
    for i in range(args.slices):
        rec.apply_ramp_filter(-np.log(stack[:, i, :]), ramp, "hann", pad, "edge")
    filter_serial = time.perf_counter() - t
    print(f"ramp filter, {args.slices} slices on one thread: {filter_serial:.2f} s")

    # -- the backend, end to end
    reconstruct(stack[:, :1], angles, center, geometry=geometry, backend="algotom_fbp", device=device)  # warm-up
    times = []
    for _ in range(args.repeats):
        t = time.perf_counter()
        reconstruct(stack, angles, center, geometry=geometry, backend="algotom_fbp", device=device)
        times.append(time.perf_counter() - t)
    best, median = min(times), float(np.median(times))
    print(f"algotom_fbp ({used}): median {median:.2f} s  best {best:.2f} s  ->  {args.slices / median:.2f} slices/s, "
          f"{updates / median / 1e9:.2f} G updates/s")
    if filter_serial and median:
        print(f"filtering is at most {min(100.0, 100 * filter_serial / median):.0f}% of that (less when it runs on several threads)")


if __name__ == "__main__":
    main()
