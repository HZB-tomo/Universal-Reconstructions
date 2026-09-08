"""
Transforms applied to a reconstructed volume after reconstruct(), before
export: currently just dtype conversion with clipping.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def to_uint16(
    volume: np.ndarray,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> tuple[np.ndarray, dict]:
    """
    Clip `volume` to [clip_min, clip_max], then linearly rescale that
    range to the full uint16 range [0, 65535].

    clip_min/clip_max default to volume.min()/volume.max(), so with no
    arguments this rescales losslessly (in the "no values were actually
    clipped" sense; uint16 quantization still applies).

    Returns (converted, record). `record` is
    {"clip_min": ..., "clip_max": ..., "source_dtype": ...}: the exact
    bounds used, worth keeping (e.g. in save_volume()'s `settings`) since
    they are needed to recover approximate physical values later.
    """
    volume = np.asarray(volume)
    source_dtype = str(volume.dtype)

    if clip_min is None:
        clip_min = float(volume.min())
    if clip_max is None:
        clip_max = float(volume.max())
    if clip_max <= clip_min:
        raise ValueError(f"clip_max ({clip_max}) must be greater than clip_min ({clip_min})")

    clipped = np.clip(volume, clip_min, clip_max)
    scaled = (clipped - clip_min) / (clip_max - clip_min) * 65535.0
    converted = np.round(scaled).astype(np.uint16)

    record = {"clip_min": clip_min, "clip_max": clip_max, "source_dtype": source_dtype}
    return converted, record
