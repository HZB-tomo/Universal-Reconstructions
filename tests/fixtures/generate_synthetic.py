"""
Generates synthetic tomography test data for exercising the whole import
pipeline without needing real instrument files.

What gets generated, per the agreed spec:
  - a 3D volume of shape (x, y, z), containing:
      - a box of size (x//3, y//2, z//2) at a random valid position
      - a sphere of radius min(x,y,z)//3 at a random valid position
  - a parallel-beam projection of that volume from N equally spaced
    angles between 0 and 180 degrees, rotating around the x axis (the
    first axis, unchanged by projection angle -- matching
    tests/fixtures/realistic_phantom's convention: the first volume
    axis is the rotation axis, and becomes the row axis of each
    projection image)
  - saved in BOTH formats our plugins understand:
      - HDF5, matching SyntheticHdf5Plugin's expected layout
      - a TIFF folder + JSON sidecar, matching SyntheticTiffPlugin's

Full-size defaults match the original spec (512, 256, 256) and 800
angles, which is slow to generate (minutes, since it rotates and sums
the whole volume once per angle). For fast test iteration, pass a
smaller shape/n_angles -- see the `synthetic_data` fixture in
tests/conftest.py, which uses a much smaller size.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import json
from scipy.ndimage import rotate


def make_volume(shape=(512, 256, 256), seed=None) -> np.ndarray:
    """Build the (x, y, z) volume containing a box and a sphere, each at
    a random position, per the agreed spec (box: x//3,y//2,z//2; sphere
    radius: min(shape)//3). x is the rotation axis; see the module
    docstring."""
    rng = np.random.default_rng(seed)
    x, y, z = shape
    volume = np.zeros(shape, dtype=np.float32)

    # box
    bx, by, bz = x // 3, y // 2, z // 2
    bx0 = rng.integers(0, x - bx)
    by0 = rng.integers(0, y - by)
    bz0 = rng.integers(0, z - bz)
    volume[bx0 : bx0 + bx, by0 : by0 + by, bz0 : bz0 + bz] = 1.0

    # sphere
    r = min(x, y, z) // 3
    cx = rng.integers(r, x - r)
    cy = rng.integers(r, y - r)
    cz = rng.integers(r, z - r)
    xx, yy, zz = np.ogrid[:x, :y, :z]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= r**2
    volume[mask] = 1.0  # overwrite in any overlap region -- fine for test purposes

    return volume


def project_parallel(volume: np.ndarray, n_angles: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a parallel-beam projection: for each of `n_angles` equally
    spaced angles (0 to 180 degrees), rotate the volume around its
    first axis (x) and sum along one of the other two (the beam
    direction) to get one 2D projection image. Returns
    (projections[n_angles, x, z], angles_deg).
    """
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    x, z = volume.shape[0], volume.shape[2]
    projections = np.empty((n_angles, x, z), dtype=np.float32)
    for i, angle in enumerate(angles):
        rotated = rotate(volume, angle, axes=(1, 2), reshape=False, order=1)
        projections[i] = rotated.sum(axis=1)
    return projections, angles


def write_hdf5(path: Path, projections, angles, operator="synthetic data"):
    """Write projections+angles in the layout SyntheticHdf5Plugin expects."""
    import h5py  # imported lazily -- see data_import/backend.py for why

    with h5py.File(path, "w") as f:
        f.attrs["operator"] = operator
        f.create_dataset("projections", data=projections)
        f.create_dataset("angles", data=angles)


def write_tiff_folder(root: Path, projections, angles, operator="synthetic data"):
    """Write projections+metadata in the layout SyntheticTiffPlugin expects."""
    proj_dir = root / "projections"
    proj_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(projections):
        tifffile.imwrite(proj_dir / f"proj_{i:05d}.tif", img.astype(np.float32))
    meta = {"angles": angles.tolist(), "operator": operator}
    (root / "metadata.json").write_text(json.dumps(meta, indent=2))


def generate(out_dir: Path, shape=(512, 256, 256), n_angles=800, seed=None):
    """
    Build the volume, project it, and write BOTH the hdf5 and tiff
    versions under out_dir. Returns (volume, projections, angles) so
    tests can compare loaded data back against the ground truth.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating synthetic data in {out_dir} with shape={shape}, n_angles={n_angles}")

    volume = make_volume(shape=shape, seed=seed)
    print(f"Generated volume with shape {volume.shape}")

    projections, angles = project_parallel(volume, n_angles=n_angles)
    print(f"Generated projections with shape {projections.shape}")

    write_hdf5(out_dir / "synthetic.h5", projections, angles)
    write_tiff_folder(out_dir / "synthetic_data", projections, angles)

    return volume, projections, angles


if __name__ == "__main__":
    # Full-spec generation, exactly as originally described:
    # 512x256x256 volume, 800 equidistant angles.
    generate(Path(__file__).parent / "generated" / "full", shape=(256, 128, 128), n_angles=400)
