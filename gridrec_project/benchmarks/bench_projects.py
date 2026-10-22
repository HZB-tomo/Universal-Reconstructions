"""
Gridrec vs FBP on real project files (HDF5), through the main URCT project's
import and preprocessing, so both reconstruct exactly the same normalized
sinograms.

    python benchmarks/bench_projects.py                       # every .h5 in ../gui_samples
    python benchmarks/bench_projects.py --files a.h5 b.h5 --slices 64
    python benchmarks/bench_projects.py --device gpu           # gridrec on a GPU (needs CuPy)

For each file: import (disk mode), normalize `--slices` rows around the middle,
find the rotation center on the middle row, then time both reconstructions and
compare them. Files that cannot be imported are reported and skipped.
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
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parents[1]))
warnings.filterwarnings("ignore")

from urct_gridrec import gridrec, make_plan  # noqa: E402


def timed(fn, repeats):
    fn()
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def prepare(path: Path, n_slices: int):
    """-> (normalized (angles, slices, width), angles_deg, center) or raises."""
    from calibration import half_scan_slice, search_center
    from core.data_interface import DataInterface
    from orchestration import PipelineRun, ProjectConfig

    answers = {"geometry_type": "parallel"}
    config = ProjectConfig(
        path=str(path), format="hdf5", storage_mode="disk", tilt_deg=0.0, completion_answers=answers,
        projections_normalized=path.name.startswith("ill"),
    )
    run = PipelineRun(config)
    run.import_data()
    try:
        run.complete()
    except Exception:  # noqa: BLE001 -- e.g. no recorded angles: assume a half turn
        config.completion_answers = {**answers, "angle_span": 180.0}
        run.complete()
    n_rows = DataInterface(run.project).shape_of("projections")[1]
    middle = n_rows // 2
    start = max(0, middle - n_slices // 2)
    config.rows = slice(start, min(n_rows, start + n_slices))
    run.preprocess()
    angles = np.asarray(DataInterface(run.project).get_metadata("angles"), dtype=float)
    half = half_scan_slice(angles)
    mid = run.normalized.shape[1] // 2
    center = search_center(run.normalized[half, mid, :], "algotom").best
    normalized = np.ascontiguousarray(run.normalized)
    run.project.close()
    return normalized, angles, center


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", type=Path)
    ap.add_argument("--samples-dir", type=Path, default=HERE.parents[1] / "gui_samples")
    ap.add_argument("--slices", type=int, default=32)
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu", "auto"])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-gb", type=float, default=6.0, help="skip files bigger than this")
    args = ap.parse_args()

    from reconstruction import Geometry, reconstruct

    files = args.files or sorted(args.samples_dir.glob("*.h5"))
    geometry = Geometry(geometry_type="parallel")
    header = (f"{'file':28s} {'angles':>6} {'slices':>6} {'width':>5} {'center':>7} | {'gridrec s/sl':>12} {'FBP s/sl':>9} "
              f"{'speedup':>8} | {'corr':>7} {'amp ratio':>9}")
    print(f"gridrec device: {args.device}")
    print(header)
    print("-" * len(header))
    for path in files:
        if path.stat().st_size > args.max_gb * 1e9:
            print(f"{path.name:28s} skipped (larger than --max-gb)")
            continue
        try:
            stack, angles, center = prepare(path, args.slices)
        except Exception as e:  # noqa: BLE001
            print(f"{path.name:28s} skipped: {type(e).__name__}: {str(e)[:70]}")
            continue
        n_angles, n_slices, width = stack.shape
        plan = make_plan(width)
        run_g = lambda: gridrec(stack, angles, center, device=args.device, plan=plan)
        run_f = lambda: reconstruct(stack, angles, center, geometry=geometry, backend="algotom_fbp", device="cpu")
        t_g = timed(run_g, args.repeats) / n_slices
        t_f = timed(run_f, args.repeats) / n_slices
        g, f = run_g(), run_f()
        mid = n_slices // 2
        corr = float(np.corrcoef(g[mid].ravel(), f[mid].ravel())[0, 1])
        ratio = float(np.abs(g[mid]).sum() / max(np.abs(f[mid]).sum(), 1e-30))
        print(f"{path.name:28s} {n_angles:>6} {n_slices:>6} {width:>5} {center:>7.2f} | {t_g:>12.4f} {t_f:>9.4f} "
              f"{t_f / t_g:>7.1f}x | {corr:>7.4f} {ratio:>9.3f}")


if __name__ == "__main__":
    main()
