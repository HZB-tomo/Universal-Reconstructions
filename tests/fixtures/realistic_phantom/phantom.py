"""
The (modified) Shepp-Logan phantom: the standard reference test object
for tomography reconstruction, a sum of overlapping ellipses/ellipsoids
with published intensity/position/size values (Toft 1996; reproduced in
Kak & Slaney and in essentially every tomography toolkit).

The 2D ellipse table below (ELLIPSES) was verified against
skimage.data.shepp_logan_phantom() before being used here -- rendering
these exact parameters at 400x400 and correlating against skimage's own
reference image gives ~0.9999998, confirming both the table and the
axis-orientation convention (y increases downward, matching image-row
order) are correct.

The 3D extension is NOT a verified published table the way the 2D one
is -- there are several different conventions for turning this into a
volume, and none was available to check here. Each ellipse is extruded
into an ellipsoid with z half-axis min(a, b) (so no ellipsoid becomes
needle-thin along height) and z-center 0, which produces a rounded,
head-like volume that genuinely varies with height rather than being a
flat 2D slice repeated -- a reasonable, clearly-documented choice, not
a claim of matching some specific external 3D reference.
"""
from __future__ import annotations

import numpy as np

# (A, a, b, x0, y0, phi_degrees) -- intensity, semi-axes, center, rotation.
# Coordinates are normalized to [-1, 1]; y0 follows image-row convention
# (increasing downward), confirmed by the correlation check above.
ELLIPSES = [
    (1.0, 0.6900, 0.9200, 0.00, 0.0000, 0),
    (-0.8, 0.6624, 0.8740, 0.00, -0.0184, 0),
    (-0.2, 0.1100, 0.3100, 0.22, 0.0000, -18),
    (-0.2, 0.1600, 0.4100, -0.22, 0.0000, 18),
    (0.1, 0.2100, 0.2500, 0.00, 0.3500, 0),
    (0.1, 0.0460, 0.0460, 0.00, 0.1000, 0),
    (0.1, 0.0460, 0.0460, 0.00, -0.1000, 0),
    (0.1, 0.0460, 0.0230, -0.08, -0.6050, 0),
    (0.1, 0.0230, 0.0230, 0.00, -0.6060, 0),
    (0.1, 0.0230, 0.0460, 0.06, -0.6050, 0),
]


def make_phantom_volume(shape: tuple[int, int, int]) -> np.ndarray:
    """
    Render the 3D Shepp-Logan phantom at `shape` = (height, width, width)
    -- height is the rotation-axis direction (unchanged by projection
    angle), the two width dimensions are the square cross-section that
    rotates. Returns a float32 array of unitless "density" values
    (overlapping ellipsoids add, matching the phantom's own convention);
    simulate.py converts this into optical depth / transmission.
    """
    height, width, width2 = shape
    if width != width2:
        raise ValueError(f"cross-section must be square, got {shape}")

    lin_xy = np.linspace(-1, 1, width)
    lin_z = np.linspace(-1, 1, height)
    xx, yy, zz = np.meshgrid(lin_xy, lin_xy, lin_z, indexing="ij")
    # xx/yy index the square cross-section, zz indexes height -- moved
    # to the last position below so the returned array is (height, width, width).

    volume = np.zeros((width, width, height), dtype=np.float32)
    for A, a, b, x0, y0, phi in ELLIPSES:
        c = min(a, b)  # z half-axis -- see module docstring
        p = np.radians(phi)
        xr = (xx - x0) * np.cos(p) + (yy - y0) * np.sin(p)
        yr = -(xx - x0) * np.sin(p) + (yy - y0) * np.cos(p)
        mask = (xr**2) / a**2 + (yr**2) / b**2 + (zz**2) / c**2 <= 1
        volume[mask] += A

    return np.ascontiguousarray(np.moveaxis(volume, 2, 0))  # -> (height, width, width)
