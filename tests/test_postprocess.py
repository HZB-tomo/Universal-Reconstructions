"""
Tests for reconstruction/postprocess.py: to_uint16()'s clip-then-rescale
behaviour and the record it returns.
"""
import numpy as np
import pytest

from reconstruction import to_uint16


def test_to_uint16_maps_the_full_range_to_0_65535_by_default():
    volume = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    converted, record = to_uint16(volume)

    assert converted.dtype == np.uint16
    assert converted[0] == 0
    assert converted[-1] == 65535
    assert record == {"clip_min": -1.0, "clip_max": 1.0, "source_dtype": "float32"}


def test_to_uint16_clips_values_outside_the_given_range():
    volume = np.array([-10.0, 0.0, 10.0], dtype=np.float32)

    converted, record = to_uint16(volume, clip_min=0.0, clip_max=5.0)

    assert converted[0] == 0       # -10 clipped down to clip_min
    assert converted[-1] == 65535  # 10 clipped up to clip_max
    assert record["clip_min"] == 0.0
    assert record["clip_max"] == 5.0


def test_to_uint16_rejects_a_degenerate_range():
    volume = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="clip_max"):
        to_uint16(volume, clip_min=1.0, clip_max=1.0)


def test_to_uint16_preserves_shape():
    volume = np.random.default_rng(0).standard_normal((3, 4, 5)).astype(np.float32)

    converted, _ = to_uint16(volume)

    assert converted.shape == volume.shape
