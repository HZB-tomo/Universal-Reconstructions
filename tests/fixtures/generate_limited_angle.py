"""
Generates a limited-angle (< 180deg) HDF5 sample using
tests/fixtures/realistic_phantom's generator, for testing the
reconstruction-rotation fallback path (reconstruction/rotation.py) on
a scan that doesn't cover a full parallel-beam period.
"""
from __future__ import annotations

from pathlib import Path

from tests.fixtures.realistic_phantom import generate, write_hdf5


def make(out_path: Path, shape=(16, 300, 300), n_angles=250, angle_span=170.0, seed=0):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    phantom = generate(
        shape=shape, n_angles=n_angles, angle_span=angle_span,
        cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0,
        noise_level=0.0, seed=seed,
    )
    write_hdf5(phantom, out_path)
    return phantom


if __name__ == "__main__":
    make(Path(__file__).parent / "generated" / "limited_angle" / "limited_angle.h5")
