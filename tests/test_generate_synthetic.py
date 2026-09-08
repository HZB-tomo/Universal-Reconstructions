"""
Tests for tests/fixtures/generate_synthetic.py's own axis convention:
the first volume axis is the rotation axis, matching
tests/fixtures/realistic_phantom's convention (previously the opposite:
the last axis was the rotation axis). See the module docstring and
doc/ARCHITECTURE.md for why this matters -- normalize_rows(),
calibration, and reconstruction all assume the row axis of a
projection stack is the untouched rotation axis.
"""
import numpy as np

from tests.fixtures.generate_synthetic import make_volume, project_parallel


def test_project_parallel_leaves_the_first_axis_untouched_by_rotation():
    """
    An object confined to one half of axis 0 must stay confined to
    that same half in every projection, at every angle -- if rotation
    were happening in the wrong plane (mixing axis 0 into the rotating
    pair), this split would blur out as the angle changes.
    """
    volume = np.zeros((20, 16, 16), dtype=np.float32)
    volume[:10, 6:10, 6:10] = 1.0  # only the first half of axis 0, off-center in the rest

    projections, angles = project_parallel(volume, n_angles=8)

    assert projections.shape == (8, 20, 16)
    assert len(angles) == 8
    for i in range(8):
        assert projections[i, :10, :].sum() > 0
        assert projections[i, 10:, :].sum() == 0


def test_make_volume_shape_and_determinism():
    volume = make_volume(shape=(20, 16, 16), seed=3)
    assert volume.shape == (20, 16, 16)
    assert volume.max() == 1.0

    again = make_volume(shape=(20, 16, 16), seed=3)
    np.testing.assert_array_equal(volume, again)
