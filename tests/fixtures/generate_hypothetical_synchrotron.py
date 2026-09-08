"""
Generates a synthetic HDF5 file matching HypotheticalSynchrotronHdf5Plugin's
expected layout, for testing that plugin without needing a real beamline
file.

Reuses make_volume() and project_parallel() from generate_synthetic.py
rather than reimplementing the object/projection logic -- only the
file-writing layout differs between vendors, so that's the only new code
here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.fixtures.generate_synthetic import make_volume, project_parallel


def make_flat_fields(shape, n_flats=10, seed=None) -> np.ndarray:
    """
    Simulate flat-field images: what the detector sees with no sample in
    the beam. Realistically these are close to a uniform bright image
    with some noise, not zero and not identical to each other -- that's
    enough for testing purposes; we don't need physical accuracy here.
    """
    rng = np.random.default_rng(seed)
    height, width = shape
    base_intensity = 1.0
    noise = rng.normal(0, 0.02, size=(n_flats, height, width))
    return (base_intensity + noise).astype(np.float32)


def write_hypothetical_synchrotron_hdf5(
    path: Path,
    projections: np.ndarray,
    flat_fields: np.ndarray,
    angles: np.ndarray,
    energy: float = 25000.0,     # eV, a plausible synchrotron X-ray energy
    pixel_size: float = 1.1,     # microns, a plausible detector pixel size
):
    """Write one hdf5 file matching HypotheticalSynchrotronHdf5Plugin's
    signature exactly -- same paths, same root attribute."""
    import h5py  # imported lazily, consistent with the rest of the codebase

    with h5py.File(path, "w") as f:
        f.attrs["beamline"] = "hypothetical_synchrotron"
        f.create_dataset("/entry/instrument/detector/data", data=projections)
        f.create_dataset("/entry/instrument/detector/flat_fields", data=flat_fields)
        f.create_dataset("/entry/instrument/sample/rotation_angle", data=angles)
        f.create_dataset("/entry/instrument/monochromator/energy", data=energy)
        f.create_dataset("/entry/instrument/detector/pixel_size", data=pixel_size)


def generate(out_path: Path, shape=(96, 64, 64), n_angles=40, n_flats=10, seed=None):
    """
    Build a volume, project it, generate matching flat fields, and write
    the whole thing to `out_path` in the hypothetical synchrotron's
    layout. Returns (volume, projections, flat_fields, angles) so tests
    can compare loaded data back against ground truth.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    volume = make_volume(shape=shape, seed=seed)
    projections, angles = project_parallel(volume, n_angles=n_angles)
    flat_fields = make_flat_fields(projections.shape[1:], n_flats=n_flats, seed=seed)

    write_hypothetical_synchrotron_hdf5(out_path, projections, flat_fields, angles)

    return volume, projections, flat_fields, angles


if __name__ == "__main__":
    generate(
        Path(__file__).parent / "generated" / "hypothetical_synchrotron.h5",
        shape=(192, 128, 128),
        n_angles=200,
    )
