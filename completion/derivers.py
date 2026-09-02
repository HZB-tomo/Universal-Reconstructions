"""
The built-in derive functions for completion's "derive" strategy
fields -- the field-specific knowledge that completion/engine.py's
generic run() loop doesn't (and shouldn't) contain. Importing this
module registers all of them via @register_deriver; see
completion/__init__.py, which imports this module for exactly that
side effect (same pattern as data_import/vendors/__init__.py importing
each vendor module to trigger their @register_plugin decorators).

Each deriver returns None if it can't compute a value yet (an input it
needs is itself still missing) -- run()'s repeat-until-no-progress loop
means a deriver that returns None this pass just gets retried on a
later pass, or a later call to run()/provide(), once its inputs become
available.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from completion.engine import register_deriver
from core.data_interface import DataInterface
from core.project import ProjectObject


@register_deriver("angle_span")
def derive_angle_span(project: ProjectObject) -> Optional[float]:
    """angle_span from angles (max - min), when angles are already known directly."""
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
    angles from angle_span + projection count + rotation_direction.
    Projection count comes from the actual data shape, never a
    separately-tracked metadata field (see core/known_fields.py's
    "angles" entry). Evenly spaced from 0, matching the convention
    already used by tests/fixtures/generate_synthetic.py and
    tests/fixtures/generate_hypothetical_synchrotron.py
    (np.linspace(0, span, n, endpoint=False)).

    rotation_direction=="cw" negates the sequence rather than reversing
    it -- the standard math convention (positive angle = counterclockwise),
    consistent with rotation_direction's own note in core/known_fields.py
    that a wrong choice here only flips orientation, never invalidates
    the reconstruction. This is a judgment call in the absence of a
    stronger convention elsewhere in the codebase -- revisit if it
    turns out to disagree with what the reconstruction stage expects.
    """
    angle_span = project.metadata.get("angle_span")
    if angle_span is None:
        return None
    direction = project.metadata.get("rotation_direction")
    if direction is None:
        return None  # rotation_direction hasn't gotten its default yet this pass -- retry later

    n_projections = DataInterface(project).shape_of("projections")[0]
    values = np.linspace(0.0, float(angle_span), n_projections, endpoint=False)
    return -values if direction == "cw" else values


@register_deriver("detector_pixel_size")
def derive_detector_pixel_size(project: ProjectObject) -> Optional[float]:
    """
    detector_pixel_size from voxel_size -- see core/known_fields.py's
    "detector_pixel_size" entry: only meaningfully distinct from
    voxel_size in cone-beam (magnification), so voxel_size is a
    reasonable stand-in until a real per-vendor magnification
    calculation is needed.
    """
    return project.metadata.get("voxel_size")
