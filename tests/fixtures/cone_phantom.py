"""
The analytic cone-beam phantom shared by the cone-beam tests: two spheres at off-centre positions along all three
axes, a rotation axis 2 detector pixels off the detector middle, and their exact line integrals for the circular
flat-detector scan described in reconstruction/backends/rtk_fdk.py. No test-skipping imports: whoever imports this
decides what to skip.
"""
import numpy as np
from scipy import ndimage as ndi

from reconstruction import Geometry

SOD, SDD, PIXEL_MM = 150.0, 300.0, 1.0
ROWS, COLS, N_ANGLES = 48, 64, 120
AXIS_COLUMN = (COLS - 1) / 2.0 + 2.0              # the rotation axis projects 2 pixels off the detector middle
SPHERES = [                                      # x, y, z (mm; z along the rotation axis), radius, mu (1/mm)
    (10.0, -5.0, 7.5, 4.0, 0.05),
    (-12.0, 6.0, -2.5, 2.5, 0.08),
]
CENTRAL_SPHERES = [                              # the same kind of scan with spheres ON the plane through the source
    (10.0, -5.0, 0.0, 6.0, 0.05),
    (-12.0, 6.0, 0.5, 3.5, 0.08),
]
CONE = Geometry(geometry_type="cone", source_to_detector_distance=SDD, source_to_object_distance=SOD,
                detector_pixel_size=PIXEL_MM * 1000.0)


def analytic_projections(axis_column=AXIS_COLUMN, spheres=SPHERES):
    """Line integrals of the spheres for the circular cone-beam scan described in rtk_fdk's docstring:
    detector axis u = (cos t, -sin t), ray r = (sin t, cos t), source at -SOD * r."""
    cc, rr = np.meshgrid(np.arange(COLS), np.arange(ROWS))
    zhat = np.array([0.0, 0.0, 1.0])
    out = np.empty((N_ANGLES, ROWS, COLS), dtype=np.float32)
    for k, theta in enumerate(np.radians(np.arange(N_ANGLES) * 360.0 / N_ANGLES)):
        u = np.array([np.cos(theta), -np.sin(theta), 0.0])
        r = np.array([np.sin(theta), np.cos(theta), 0.0])
        source = -SOD * r
        detector_centre = (SDD - SOD) * r
        point = (detector_centre[None, None, :] + ((cc - axis_column) * PIXEL_MM)[..., None] * u
                 + ((rr - (ROWS - 1) / 2) * PIXEL_MM)[..., None] * zhat)
        ray = point - source
        length = np.linalg.norm(ray, axis=-1)
        total = np.zeros((ROWS, COLS))
        for x0, y0, z0, radius, mu in spheres:
            d = np.linalg.norm(np.cross((np.array([x0, y0, z0]) - source)[None, None, :], ray), axis=-1) / length
            total += mu * 2 * np.sqrt(np.clip(radius ** 2 - d ** 2, 0, None))
        out[k] = total
    return out


def voxel_size_mm():
    return PIXEL_MM * SOD / SDD


def blobs(volume, threshold_fraction=0.4):
    """Centroids, in mm in the scan's (x, y, z) frame, of the connected blobs above a threshold."""
    labelled, count = ndi.label(volume > threshold_fraction * volume.max())
    found = []
    for label in range(1, count + 1):
        mask = labelled == label
        if mask.sum() < 20:
            continue
        index = np.argwhere(mask)
        weight = volume[mask]
        iz, iy, ix = (index * weight[:, None]).sum(0) / weight.sum()
        vox = voxel_size_mm()
        nz, ny, nx = volume.shape
        found.append(((ix - (nx - 1) / 2) * vox, (iy - (ny - 1) / 2) * vox, (iz - (nz - 1) / 2) * vox, int(mask.sum())))
    return sorted(found, key=lambda b: -b[3])
