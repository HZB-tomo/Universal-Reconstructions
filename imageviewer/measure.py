"""
The arithmetic of measuring on an image, without Qt. Points are (x, y) in the canvas's
view coordinates: x runs along the columns and y along the rows, and a pixel (row r,
column c) covers x in [c, c+1], y in [r, r+1] (so its centre is at c + 0.5, r + 0.5).
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np


def line_length(p0, p1) -> float:
    """Length in pixels."""
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1])


def line_angle(p0, p1) -> float:
    """Angle in degrees in (-180, 180]: 0 points right and positive turns counter-clockwise
    *as seen on screen*, where the image's rows run downward."""
    return math.degrees(math.atan2(-(p1[1] - p0[1]), p1[0] - p0[0]))


def describe_line(p0, p1, pixel: Optional[tuple] = None) -> str:
    """One line of text for a label: length (also in a physical unit when `pixel` = (size, unit)) and angle."""
    length = line_length(p0, p1)
    text = f"{length:.1f} px"
    if pixel:
        size, unit = pixel
        text += f" = {length * size:.4g} {unit}"
    angle = round(line_angle(p0, p1), 1) + 0.0          # + 0.0 turns -0.0 into 0.0
    return f"{text}, {angle:.1f}°"


def line_profile(frame: np.ndarray, p0, p1, width: float = 1.0, spacing: float = 1.0):
    """Values of `frame` along the segment p0 -> p1, one sample per `spacing` pixels (bilinear).
    `width` > 1 averages that many parallel lines across the line's width. Samples outside the
    frame are NaN. Returns (distance from p0 in pixels, values)."""
    frame = np.asarray(frame, dtype=np.float64)
    length = line_length(p0, p1)
    n = max(2, int(math.ceil(length / spacing)) + 1)
    t = np.linspace(0.0, 1.0, n)
    xs = p0[0] + (p1[0] - p0[0]) * t
    ys = p0[1] + (p1[1] - p0[1]) * t
    n_lines = max(1, int(round(width)))
    if n_lines > 1 and length > 0:
        nx, ny = -(p1[1] - p0[1]) / length, (p1[0] - p0[0]) / length          # unit normal
        offsets = np.linspace(-(n_lines - 1) / 2.0, (n_lines - 1) / 2.0, n_lines)
    else:
        nx = ny = 0.0
        offsets = np.zeros(1)
    from scipy.ndimage import map_coordinates

    rows = (ys[None, :] + offsets[:, None] * ny) - 0.5            # pixel centres sit at +0.5
    cols = (xs[None, :] + offsets[:, None] * nx) - 0.5
    samples = map_coordinates(frame, [rows.ravel(), cols.ravel()], order=1, mode="constant", cval=np.nan)
    samples = samples.reshape(len(offsets), n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # an all-NaN column (line leaves the frame)
        values = np.nanmean(samples, axis=0)
    return t * length, values
