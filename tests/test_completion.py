"""
Tests for completion/: the automatic pass (run()) and recording
externally-supplied answers (provide()). No files involved -- these
build a bare ProjectObject by hand, the same way tests/test_project.py
builds bare MetadataBundles. Built-in derivers (completion/derivers.py)
are covered in tests/test_derivers.py.
"""
import numpy as np
import pytest

import completion
from core.handles import RamHandle
from core.metadata import MetadataBundle
from core.project import ProjectObject


def _bare_project(n_angles=10):
    meta = MetadataBundle()
    meta.set("projections_path", "/projections", tier="mandatory_hard")
    # A real (if tiny) projections handle -- the "angles" deriver reads
    # the projection count off actual data shape, not a metadata field
    # (see core/known_fields.py's "angles" entry), so a bare metadata-only
    # project isn't representative of what completion.run() actually
    # sees once a real import has happened.
    handles = {"projections": RamHandle(np.zeros((n_angles, 2, 2), dtype=np.float32))}
    return ProjectObject(format="hdf5", storage_mode="ram", metadata=meta, data_handles=handles)


def test_run_applies_defaults():
    project = _bare_project()

    completion.run(project)

    assert project.metadata.get("voxel_size") == 1.0
    assert project.metadata.get_field("voxel_size").source == "default"


def test_run_never_touches_computed_fields():
    project = _bare_project()

    completion.run(project)

    field = project.metadata.get_field("center_of_rotation")
    assert field.is_missing  # calibration's job, not completion's


def test_run_leaves_ask_only_fields_missing():
    project = _bare_project()

    completion.run(project)

    assert "geometry_type" in project.missing_soft_fields()


def test_provide_records_user_value_and_survives_a_rerun():
    project = _bare_project()

    completion.provide(project, "geometry_type", "cone")

    field = project.metadata.get_field("geometry_type")
    assert field.value == "cone"
    assert field.source == "user"

    completion.run(project)  # re-running shouldn't clobber the user's answer
    assert project.metadata.get("geometry_type") == "cone"


def test_provide_unlocks_conditional_field():
    project = _bare_project()
    assert project.metadata.get_field("source_to_detector_distance") is None

    completion.provide(project, "geometry_type", "cone")

    field = project.metadata.get_field("source_to_detector_distance")
    assert field is not None
    assert field.is_missing  # now flagged as relevant, still needs an answer


def test_provide_rejects_unknown_key():
    project = _bare_project()

    with pytest.raises(ValueError):
        completion.provide(project, "not_a_real_field", 1)
