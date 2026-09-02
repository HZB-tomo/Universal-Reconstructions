"""
Tests for core/known_fields.py: reconcile() closing the "plugin forgot
to mention it" gap, its conditional-field behavior, and status()
giving a full readout. No files involved -- these build MetadataBundles
by hand, the same way tests/test_project.py does.
"""
from core.known_fields import KNOWN_FIELDS, reconcile, status
from core.metadata import MetadataBundle


def test_reconcile_fills_in_fields_the_plugin_never_mentioned():
    bundle = MetadataBundle()
    bundle.set("projections_path", "/projections", tier="mandatory_hard")
    assert bundle.get_field("voxel_size") is None  # plugin never touched this at all

    reconcile(bundle)

    field = bundle.get_field("voxel_size")
    assert field is not None
    assert field.is_missing
    assert field.tier == "mandatory_soft"


def test_reconcile_does_not_touch_fields_the_plugin_already_set():
    bundle = MetadataBundle()
    bundle.set("voxel_size", 3.5, tier="mandatory_soft", source="file")

    reconcile(bundle)

    field = bundle.get_field("voxel_size")
    assert field.value == 3.5
    assert field.source == "file"


def test_reconcile_skips_conditional_field_until_condition_is_known():
    bundle = MetadataBundle()
    bundle.declare_missing("geometry_type", tier="mandatory_soft")

    reconcile(bundle)
    assert bundle.get_field("source_to_detector_distance") is None

    bundle.set("geometry_type", "cone", tier="mandatory_soft", source="user")
    reconcile(bundle)

    field = bundle.get_field("source_to_detector_distance")
    assert field is not None
    assert field.is_missing


def test_reconcile_never_flags_conditional_field_for_parallel_beam():
    bundle = MetadataBundle()
    bundle.set("geometry_type", "parallel", tier="mandatory_soft", source="user")

    reconcile(bundle)

    assert bundle.get_field("source_to_detector_distance") is None
    assert bundle.get_field("source_to_object_distance") is None


def test_status_reports_every_known_field():
    bundle = MetadataBundle()

    rows = status(bundle)

    assert {row.key for row in rows} == set(KNOWN_FIELDS)
    assert all(not row.resolved for row in rows)


def test_status_reflects_resolved_fields():
    bundle = MetadataBundle()
    bundle.set("voxel_size", 2.0, tier="mandatory_soft", source="file")

    rows = {row.key: row for row in status(bundle)}

    assert rows["voxel_size"].resolved
    assert rows["voxel_size"].value == 2.0
    assert not rows["geometry_type"].resolved
