"""
Built-in derive functions for completion's "derive" strategy fields.
Importing this module registers each via @register_deriver.

A deriver returns None if it cannot compute a value yet (an input it
needs is itself still missing); run()'s repeat-until-no-progress loop
retries it on a later pass.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from completion.engine import register_deriver
from core.data_interface import DataInterface
from core.project import ProjectObject


@register_deriver("angle_span")
def derive_angle_span(project: ProjectObject) -> Optional[float]:
    """angle_span from angles (max - min), when angles are already known."""
    angles = project.metadata.get("angles")
    if angles is None:
        return None
    angles = np.asarray(angles, dtype=float)
    if angles.size == 0:
        return None
    return float(angles.max() - angles.min())


@register_deriver("angles")
def derive_angles(project: ProjectObject) -> Optional[np.ndarray]:
    """
    angles from angle_span, projection count, and rotation_direction.
    Projection count comes from the data shape, not a separate
    metadata field. Evenly spaced from 0
    (np.linspace(0, span, n, endpoint=False)).

    rotation_direction=="cw" negates the sequence rather than reversing
    it: positive angle is counterclockwise.
    """
    angle_span = project.metadata.get("angle_span")
    if angle_span is None:
        return None
    direction = project.metadata.get("rotation_direction")
    if direction is None:
        return None  # rotation_direction hasn't received its default yet this pass

    n_projections = DataInterface(project).shape_of("projections")[0]
    values = np.linspace(0.0, float(angle_span), n_projections, endpoint=False)
    return -values if direction == "cw" else values


@register_deriver("detector_pixel_size")
def derive_detector_pixel_size(project: ProjectObject) -> Optional[float]:
    """detector_pixel_size from voxel_size. Distinct from voxel_size
    only in cone-beam (magnification)."""
    return project.metadata.get("voxel_size")
