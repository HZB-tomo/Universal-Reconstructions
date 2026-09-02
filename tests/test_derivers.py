"""
Tests for completion/derivers.py's built-in derive functions, and for
completion.run()'s repeat-until-no-progress loop that makes cross-field
dependencies (angles needs rotation_direction, which only gets its
default applied during the same run()) actually resolve in one call.
"""
import numpy as np
import pytest

import completion
from core.handles import RamHandle
from core.metadata import MetadataBundle
from core.project import ProjectObject


def _project(n_angles=10):
    meta = MetadataBundle()
    meta.set("projections_path", "/projections", tier="mandatory_hard")
    handles = {"projections": RamHandle(np.zeros((n_angles, 2, 2), dtype=np.float32))}
    return ProjectObject(format="hdf5", storage_mode="ram", metadata=meta, data_handles=handles)


def test_angle_span_derives_from_angles():
    project = _project()
    project.metadata.set("angles", np.array([0.0, 30.0, 60.0, 90.0]), tier="mandatory_soft", source="file")

    completion.run(project)

    assert project.metadata.get("angle_span") == 90.0
    assert project.metadata.get_field("angle_span").source == "derived"


def test_angles_derives_from_angle_span_and_default_rotation_direction():
    """
    angles needs angle_span AND rotation_direction, but rotation_direction
    only gets its default ("ccw") applied during the SAME run() call --
    this only works if run() keeps looping until nothing changes, not
    just a single pass over the field list.
    """
    project = _project(n_angles=4)
    project.metadata.set("angle_span", 180.0, tier="mandatory_soft", source="user")

    completion.run(project)

    angles = project.metadata.get("angles")
    assert angles is not None
    np.testing.assert_allclose(angles, [0.0, 45.0, 90.0, 135.0])
    assert project.metadata.get_field("angles").source == "derived"


def test_angles_respects_explicit_clockwise_direction():
    project = _project(n_angles=4)
    project.metadata.set("angle_span", 180.0, tier="mandatory_soft", source="user")
    project.metadata.set("rotation_direction", "cw", tier="optional", source="user")

    completion.run(project)

    angles = project.metadata.get("angles")
    np.testing.assert_allclose(angles, [0.0, -45.0, -90.0, -135.0])


def test_angles_and_angle_span_both_stay_missing_if_neither_is_ever_given():
    """Circular dependency with no external input at all: run() should
    make no progress on either and leave both missing, not loop forever."""
    project = _project()

    completion.run(project)

    assert project.metadata.get_field("angles").is_missing
    assert project.metadata.get_field("angle_span").is_missing


def test_providing_angle_span_derives_angles_via_provide():
    project = _project(n_angles=5)

    completion.provide(project, "angle_span", 100.0)

    angles = project.metadata.get("angles")
    assert angles is not None
    assert len(angles) == 5


def test_detector_pixel_size_derives_from_voxel_size():
    project = _project()
    project.metadata.set("voxel_size", 2.5, tier="mandatory_soft", source="file")

    completion.run(project)

    assert project.metadata.get("detector_pixel_size") == 2.5
    assert project.metadata.get_field("detector_pixel_size").source == "derived"


def test_detector_pixel_size_falls_back_to_default_without_voxel_size():
    """voxel_size and detector_pixel_size resolve in the same run() pass
    (voxel_size defaults first), so this checks the fallback still works
    when voxel_size itself has no value to derive from -- it shouldn't,
    since voxel_size always gets its own default applied first; this
    guards against that assumption silently breaking later."""
    project = _project()

    completion.run(project)

    # voxel_size defaults to 1.0, so detector_pixel_size derives from
    # that rather than needing its own static default.
    assert project.metadata.get("voxel_size") == 1.0
    assert project.metadata.get("detector_pixel_size") == 1.0
    assert project.metadata.get_field("detector_pixel_size").source == "derived"


def test_hypothetical_synchrotron_style_project_completes_angle_span_automatically():
    """
    Regression check against a realistic case: a vendor (like
    hypothetical_synchrotron_hdf5) that provides angles directly should
    get angle_span derived automatically now, where before this needed
    a manual completion.provide() call.
    """
    project = _project(n_angles=8)
    project.metadata.set("angles", np.linspace(0, 174, 8), tier="mandatory_soft", source="file")

    completion.run(project)

    assert project.metadata.get("angle_span") == pytest.approx(174.0)
    assert not project.metadata.get_field("angle_span").is_missing
