"""
Tests for ProjectObject's readiness-check methods and for
MetadataBundle's overwrite-without-history behaviour for computed
fields. These don't touch any files at all -- pure in-memory checks of
core/project.py and core/metadata.py.
"""
from core.project import ProjectObject
from core.metadata import MetadataBundle


def test_is_ready_for_import_true_when_no_hard_missing():
    meta = MetadataBundle()
    meta.set("projections_path", "/projections", tier="mandatory_hard")
    project = ProjectObject(format="hdf5", storage_mode="ram", metadata=meta)
    assert project.is_ready_for_import()


def test_is_ready_for_import_false_when_hard_missing():
    meta = MetadataBundle()
    meta.declare_missing("projections_path", tier="mandatory_hard")
    project = ProjectObject(format="hdf5", storage_mode="ram", metadata=meta)
    assert not project.is_ready_for_import()
    assert "projections_path" in project.missing_hard_fields()


def test_needs_user_input_when_soft_missing():
    meta = MetadataBundle()
    meta.set("projections_path", "/projections", tier="mandatory_hard")
    meta.declare_missing("voxel_size", tier="mandatory_soft")
    project = ProjectObject(format="hdf5", storage_mode="ram", metadata=meta)
    assert project.is_ready_for_import()
    assert project.needs_user_input()
    assert "voxel_size" in project.missing_soft_fields()


def test_computed_field_overwrite_no_history():
    meta = MetadataBundle()
    meta.set("center_of_rotation", 100.0, tier="computed", source="file")
    meta.set_computed("center_of_rotation", 102.5)
    field = meta.get_field("center_of_rotation")
    assert field.value == 102.5
    assert field.source == "computed"
