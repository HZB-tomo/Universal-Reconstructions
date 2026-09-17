"""
Generates a synthetic BAMline on-the-fly HDF5 file: a leading
flat-field plateau (rotation stage stationary), a rotation ramp that
deliberately overshoots 180 degrees (room for the motor to decelerate),
and a trailing plateau -- the exact shape
data_import/vendors/bamline.py's window-detection logic is built to
handle. Also seeds a couple of *other* rotation-table NDAttributes with
pure encoder noise, to exercise picking the real one by value range
rather than by name (see the plugin's module docstring).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import rotate

from tests.fixtures.generate_synthetic import make_volume

# Every unused-rotation-table candidate the plugin knows about, minus
# whichever one this generate() call designates as the real channel.
_ALL_ANGLE_PATHS = (
    "/entry/instrument/NDAttributes/SAMPLE_MICOS_W1",
    "/entry/instrument/NDAttributes/SAMPLE_MICOS_W2",
    "/entry/instrument/NDAttributes/CT_MICOS_W",
    "/entry/instrument/NDAttributes/SAMPLE_HUBER_W",
    "/entry/instrument/NDAttributes/AEROTECH_W",
    "/entry/instrument/NDAttributes/SAMPLE_W",
)


def _project_at_angles(volume: np.ndarray, angles_deg) -> np.ndarray:
    x, z = volume.shape[0], volume.shape[2]
    projections = np.empty((len(angles_deg), x, z), dtype=np.float32)
    for i, angle in enumerate(angles_deg):
        rotated = rotate(volume, angle, axes=(1, 2), reshape=False, order=1)
        projections[i] = rotated.sum(axis=1)
    return projections


def generate(
    path: Path,
    shape=(48, 32, 32),
    n_flats=15,
    n_rotate=200,
    overshoot_deg=5.0,
    n_trailing=10,
    angle_attribute="/entry/instrument/NDAttributes/SAMPLE_MICOS_W2",
    seed=None,
):
    """
    Writes one BAMline-shaped hdf5 file at `path`. Returns (volume,
    n_flats, n_rotate, overshoot_deg) -- the fixture parameters a test
    needs to independently compute the expected projection window,
    without calling into the plugin itself.
    """
    import h5py  # imported lazily -- see data_import/backend.py for why

    rng = np.random.default_rng(seed)
    volume = make_volume(shape=shape, seed=seed)

    span = 180.0 + overshoot_deg
    # Starts at one step past 0, not at 0 itself: a real motor has
    # already moved a resolvable amount by its first captured frame
    # after leaving the plateau (see the real angle list this plugin
    # was built from -- frame 0 is 0.0, frame 1 is already 0.0145, never
    # a second frame sitting at exactly the plateau's value).
    rotate_angles = (np.arange(1, n_rotate + 1) / n_rotate) * span

    rotating_frames = _project_at_angles(volume, rotate_angles)
    leading_frames = np.broadcast_to(rotating_frames[0], (n_flats, shape[0], shape[2])).copy()
    trailing_frames = np.broadcast_to(rotating_frames[-1], (n_trailing, shape[0], shape[2])).copy()
    all_frames = np.concatenate([leading_frames, rotating_frames, trailing_frames], axis=0).astype(np.float32)

    leading_angles = np.zeros(n_flats)
    trailing_angles = np.full(n_trailing, rotate_angles[-1])
    full_angles = np.concatenate([leading_angles, rotate_angles, trailing_angles])
    n_total = len(full_angles)

    with h5py.File(path, "w") as f:
        f.create_dataset("/entry/data/data", data=all_frames)
        f.create_dataset("/entry/instrument/NDAttributes/RingCurrent", data=np.full(n_total, 300.0))
        f.create_dataset("/entry/instrument/NDAttributes/Binning_X", data=np.full(n_total, 1))
        # Per-frame NDAttributes in real files, not scalars -- constant
        # across the scan, same as a real one would be, to catch a
        # plugin that reads the whole array where a scalar is wanted.
        f.create_dataset("/entry/instrument/NDAttributes/CT_Pixelsize", data=np.full(n_total, 0.65))
        f.create_dataset("/entry/instrument/NDAttributes/DMM_Energy", data=np.full(n_total, 30000.0))
        f.create_dataset(angle_attribute, data=full_angles)
        # A couple of OTHER rotation-table attributes, unused for this
        # scan -- pure noise, exactly like a real BAMline file's idle
        # stages.
        for unused_path in _ALL_ANGLE_PATHS:
            if unused_path != angle_attribute:
                f.create_dataset(unused_path, data=rng.uniform(-1e-4, 1e-4, size=n_total))

    return volume, n_flats, n_rotate, overshoot_deg
