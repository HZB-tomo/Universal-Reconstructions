"""
Shared helpers for the cone-beam FDK backends (rtk_fdk, astra_fdk): the same
geometry checks and the same transmission -> line-integral conversion, so both
refuse bad input with the same messages. Not a backend itself -- no @register_backend.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from reconstruction.backend import Geometry


class ConeParameters(NamedTuple):
    sdd: float            # source to detector, mm
    sod: float            # source to rotation axis, mm
    pixel_mm: float       # detector pitch
    voxel_mm: float       # cubic voxel: the detector pixel at the axis
    axis_column: float    # detector column onto which the rotation axis projects


def cone_parameters(geometry: Geometry, center) -> ConeParameters:
    """Validate a cone geometry and the single center of rotation; raises ValueError."""
    sdd, sod, pixel_um = (
        geometry.source_to_detector_distance, geometry.source_to_object_distance, geometry.detector_pixel_size,
    )
    missing = [name for name, value in (
        ("source_to_detector_distance", sdd), ("source_to_object_distance", sod), ("detector_pixel_size", pixel_um),
    ) if not value]
    if missing:
        raise ValueError(f"cone-beam reconstruction needs {', '.join(missing)}")
    if not float(sdd) > float(sod) > 0:
        raise ValueError(f"need source_to_detector_distance ({sdd}) > source_to_object_distance ({sod}) > 0")

    if center is None:
        raise ValueError("cone-beam reconstruction needs the center of rotation (the detector column the axis projects onto)")
    centers = np.atleast_1d(np.asarray(center, dtype=np.float64))
    if np.ptp(centers) > 1e-6:
        raise ValueError("cone-beam reconstruction takes one center of rotation; a per-slice (tilt) correction is not supported")

    pixel_mm = float(pixel_um) / 1000.0
    return ConeParameters(float(sdd), float(sod), pixel_mm, pixel_mm / (float(sdd) / float(sod)), float(centers[0]))


def line_integrals(sinograms: np.ndarray) -> np.ndarray:
    """-log of transmission (values <= 0 count as 1, like the other backends); float32, C-contiguous."""
    transmission = np.array(sinograms, dtype=np.float32)
    transmission[transmission <= 0.0] = 1.0
    return np.ascontiguousarray(-np.log(transmission), dtype=np.float32)


def output_size(n_rows: int, n_cols: int, size_xy: Optional[int], size_z: Optional[int]) -> tuple:
    """(voxels across, voxels along the axis): the detector's width and number of rows unless given."""
    return int(size_xy or n_cols), int(size_z or n_rows)
