"""
Cone-beam check on a 3D ellipsoid (Shepp-Logan style) phantom, using the GPU.

  * the phantom's cone-beam projections are computed analytically (exact ray/ellipsoid chords) on the
    GPU with CuPy, in the geometry documented in doc/CONE_BEAM.md;
  * they are reconstructed with the app's `rtk_fdk` backend (CPU) through reconstruction.reconstruct();
  * and with the app's `astra_fdk` backend (GPU), through the same call;
  * (a hand-written ASTRA FDK_CUDA call is kept as an independent check of the backend's geometry);
  * both volumes are compared against the voxelized ground truth.

    python benchmarks/gpu_cone_phantom.py [--size 128] [--angles 360] [--center-offset 3.0]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction import Geometry, reconstruct, resolve_device  # noqa: E402

SOD, SDD = 150.0, 300.0     # mm; magnification 2
PIXEL_MM = 1.0              # detector pitch

# centre (x, y, z) mm, semi-axes (a, b, c) mm, rotation about z (deg), mu (1/mm); values add up where they overlap
ELLIPSOIDS = [
    ((0, 0, 0), (36, 46, 40), 0, 0.020),
    ((0, 0, 0), (33, 43, 37), 0, -0.010),
    ((11, 0, -4), (8, 22, 14), -18, 0.012),
    ((-11, 0, -4), (11, 30, 16), 18, 0.012),
    ((0, 18, 4), (14, 12, 12), 0, 0.016),
    ((0, -3, 0), (4, 4, 4), 0, 0.020),
    ((0, -18, 12), (4, 3, 8), 0, 0.020),
    ((-12, -14, -20), (4, 6, 5), 30, 0.025),
    ((8, 6, 24), (5, 5, 5), 0, 0.030),      # off the central plane: shows the cone-beam artifacts
]


def _frame(rot_deg):
    t = np.radians(rot_deg)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


def project(n_rows, n_cols, angles_deg, axis_column):
    """Exact line integrals (mm * 1/mm) of the ellipsoids, the app's geometry. -> (angles, rows, cols) float32."""
    cc, rr = cp.meshgrid(cp.arange(n_cols, dtype=cp.float64), cp.arange(n_rows, dtype=cp.float64))
    out = np.empty((len(angles_deg), n_rows, n_cols), dtype=np.float32)
    for k, theta in enumerate(np.radians(angles_deg)):
        u = np.array([np.cos(theta), -np.sin(theta), 0.0])
        r = np.array([np.sin(theta), np.cos(theta), 0.0])
        src = -SOD * r
        centre = (SDD - SOD) * r
        pt = (cp.asarray(centre)[None, None, :] + ((cc - axis_column) * PIXEL_MM)[..., None] * cp.asarray(u)
              + ((rr - (n_rows - 1) / 2) * PIXEL_MM)[..., None] * cp.asarray([0.0, 0.0, 1.0]))
        d = pt - cp.asarray(src)
        d /= cp.linalg.norm(d, axis=-1, keepdims=True)
        total = cp.zeros((n_rows, n_cols))
        for c, ax, rot, mu in ELLIPSOIDS:
            m = cp.asarray(_frame(rot).T / np.array(ax)[:, None])      # world offset -> unit-sphere frame
            o = m @ cp.asarray(src - np.array(c, dtype=float))
            dd = d @ m.T
            a = (dd * dd).sum(-1)
            b = (dd * o).sum(-1)
            disc = b * b - a * ((o * o).sum() - 1.0)
            total += mu * 2.0 * cp.sqrt(cp.clip(disc, 0, None)) / a
        out[k] = cp.asnumpy(total)
    return out


def voxelize(n_z, n_xy, voxel):
    """Ground truth on the app's output grid (slices, y, x), mm positions as in tests/test_cone_beam.py."""
    z, y, x = cp.meshgrid(*[(cp.arange(n, dtype=cp.float64) - (n - 1) / 2) * voxel for n in (n_z, n_xy, n_xy)], indexing="ij")
    pts = cp.stack([x, y, z], -1)
    vol = cp.zeros((n_z, n_xy, n_xy))
    for c, ax, rot, mu in ELLIPSOIDS:
        q = (pts - cp.asarray(c, dtype=cp.float64)) @ cp.asarray(_frame(rot) / np.array(ax)[None, :])
        vol += mu * ((q * q).sum(-1) <= 1.0)
    return cp.asnumpy(vol).astype(np.float32)


def astra_fdk(line_integrals, angles_deg, axis_column, n_xy, n_z, voxel):
    """ASTRA's CUDA FDK in the same geometry. -> volume in the app's (slice, y, x) orientation, 1/mm."""
    import astra
    n_ang, n_rows, n_cols = line_integrals.shape
    vec = np.zeros((n_ang, 12))
    for k, theta in enumerate(np.radians(angles_deg)):
        u = np.array([np.cos(theta), -np.sin(theta), 0.0])
        r = np.array([np.sin(theta), np.cos(theta), 0.0])
        vec[k, 0:3] = -SOD * r
        vec[k, 3:6] = (SDD - SOD) * r + ((n_cols - 1) / 2 - axis_column) * PIXEL_MM * u
        vec[k, 6:9] = PIXEL_MM * u
        vec[k, 9:12] = [0.0, 0.0, PIXEL_MM]
    pg = astra.create_proj_geom("cone_vec", n_rows, n_cols, vec)
    hx, hz = n_xy * voxel / 2, n_z * voxel / 2
    vg = astra.create_vol_geom(n_xy, n_xy, n_z, -hx, hx, -hx, hx, -hz, hz)
    pid = astra.data3d.create("-proj3d", pg, np.ascontiguousarray(line_integrals.transpose(1, 0, 2)))
    vid = astra.data3d.create("-vol", vg, 0.0)
    cfg = astra.astra_dict("FDK_CUDA")
    cfg["ReconstructionDataId"], cfg["ProjectionDataId"] = vid, pid
    cfg["option"] = {"FilterType": "ram-lak"}
    alg = astra.algorithm.create(cfg)
    astra.algorithm.run(alg)
    vol = astra.data3d.get(vid)
    astra.algorithm.delete(alg), astra.data3d.delete(pid), astra.data3d.delete(vid)
    return vol


def nrmse(vol, truth, mask):
    return float(np.linalg.norm((vol - truth)[mask]) / np.linalg.norm(truth[mask]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128, help="detector rows = columns")
    ap.add_argument("--angles", type=int, default=360)
    ap.add_argument("--center-offset", type=float, default=3.0, help="rotation axis offset from the detector middle (pixels)")
    ap.add_argument("--skip-rtk", action="store_true")
    args = ap.parse_args()

    n = args.size
    angles = np.arange(args.angles) * 360.0 / args.angles
    axis_column = (n - 1) / 2 + args.center_offset
    voxel = PIXEL_MM * SOD / SDD
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}  detector {n}x{n}, {args.angles} angles, "
          f"axis column {axis_column}")

    t = time.perf_counter()
    li = project(n, n, angles, axis_column)
    cp.cuda.Stream.null.synchronize()
    print(f"analytic projection on GPU: {time.perf_counter() - t:.2f} s, max line integral {li.max():.3f}")
    truth = voxelize(n, n, voxel)
    zz, yy, xx = np.mgrid[:n, :n, :n]
    inside = ((xx - (n - 1) / 2) ** 2 + (yy - (n - 1) / 2) ** 2 + (zz - (n - 1) / 2) ** 2) < (0.45 * n) ** 2

    geometry = Geometry(geometry_type="cone", source_to_detector_distance=SDD, source_to_object_distance=SOD,
                        detector_pixel_size=PIXEL_MM * 1000.0)
    results = {}
    t = time.perf_counter()
    results["raw ASTRA FDK_CUDA"] = (astra_fdk(li, angles, axis_column, n, n, voxel), time.perf_counter() - t)
    t = time.perf_counter()
    vol = reconstruct(np.exp(-li).astype(np.float32), angles, axis_column, geometry=geometry, backend="astra_fdk")
    results["app astra_fdk (GPU)"] = (vol / 10.0, time.perf_counter() - t)
    if not args.skip_rtk:
        print(f"rtk_fdk device resolves to: {resolve_device('rtk_fdk', 'gpu')}")
        t = time.perf_counter()
        vol = reconstruct(np.exp(-li).astype(np.float32), angles, axis_column, geometry=geometry, backend="rtk_fdk")
        results["app rtk_fdk (CPU)"] = (vol / 10.0, time.perf_counter() - t)     # the app returns 1/cm; compare in 1/mm

    mid = n // 2
    print(f"\nground truth: max {truth.max():.4f} 1/mm; central-slice mean {truth[mid][inside[mid]].mean():.4f}")
    for name, (vol, secs) in results.items():
        flips = {"as is": vol, "y flipped": vol[:, ::-1, :]}
        label, best = min(flips.items(), key=lambda kv: nrmse(kv[1], truth, inside))
        print(f"{name:24s} {secs:6.2f} s  NRMSE all {nrmse(best, truth, inside):.3f}  central slice "
              f"{nrmse(best[mid], truth[mid], inside[mid]):.3f}  orientation: {label}  "
              f"max {best.max():.4f}  corr {np.corrcoef(best[inside], truth[inside])[0, 1]:.4f}")
    names = list(results)
    for other in names[1:]:
        print(f"{other} vs {names[0]}: NRMSE {nrmse(results[other][0], results[names[0]][0], inside):.3f}")


if __name__ == "__main__":
    main()
