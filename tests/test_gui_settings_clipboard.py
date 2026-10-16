"""
Tests for gui/settings_clipboard.py: copying ProjectConfig/
ProjectSession settings from one project to another. Pure Python, no
Qt/QApplication needed -- ProjectPanel just calls these two functions.
"""
from orchestration.config import ProjectConfig
from gui.session import ProjectSession
from gui.settings_clipboard import ALL_COPYABLE_FIELDS, COPYABLE_SETTINGS, copy_settings, paste_settings


def _session(**config_kwargs) -> ProjectSession:
    return ProjectSession(config=ProjectConfig(path="scan.h5", **config_kwargs))


def test_all_copyable_fields_matches_the_category_lists():
    flattened = [name for fields in COPYABLE_SETTINGS.values() for name, _ in fields]
    assert ALL_COPYABLE_FIELDS == flattened


def test_copy_settings_reads_config_fields():
    session = _session(center=41.5, tilt_deg=0.3, rotate_deg=12.0, dtype="uint16")

    values = copy_settings(session, ["center", "tilt_deg", "rotate_deg", "dtype"])

    assert values == {"center": 41.5, "tilt_deg": 0.3, "rotate_deg": 12.0, "dtype": "uint16"}


def test_copy_settings_reads_crop_yx_from_the_session_not_config():
    session = _session()
    session.export_crop_yx = (10, 20, 30, 40)

    values = copy_settings(session, ["crop_yx"])

    assert values == {"crop_yx": (10, 20, 30, 40)}


def test_copy_settings_copies_dict_fields_instead_of_aliasing():
    session = _session()
    session.config.reconstruction_options["num_iter"] = 50

    values = copy_settings(session, ["reconstruction_options"])
    values["reconstruction_options"]["num_iter"] = 999

    assert session.config.reconstruction_options["num_iter"] == 50


def test_paste_settings_writes_config_fields():
    source = _session(center=41.5, tilt_deg=0.3)
    target = _session(center=None, tilt_deg=None)

    paste_settings(target, copy_settings(source, ["center", "tilt_deg"]))

    assert target.config.center == 41.5
    assert target.config.tilt_deg == 0.3


def test_paste_settings_writes_crop_yx_onto_the_session_not_config():
    source = _session()
    source.export_crop_yx = (10, 20, 30, 40)
    target = _session()

    paste_settings(target, copy_settings(source, ["crop_yx"]))

    assert target.export_crop_yx == (10, 20, 30, 40)
    assert not hasattr(target.config, "crop_yx")


def test_pasting_the_same_clipboard_onto_two_targets_gives_each_an_independent_dict():
    source = _session()
    source.config.ring_options["sigma"] = 3
    clipboard = copy_settings(source, ["ring_options"])

    target_a, target_b = _session(), _session()
    paste_settings(target_a, clipboard)
    paste_settings(target_b, clipboard)
    target_a.config.ring_options["sigma"] = 999

    assert target_b.config.ring_options["sigma"] == 3
    assert clipboard["ring_options"]["sigma"] == 3


def test_geometry_and_identity_fields_are_not_copyable():
    # per the feature request: only things that "usually don't come
    # from the project file" are copyable -- project_id/path/format/
    # completion_answers (geometry_type, angle_span, voxel_size, ...)
    # must never appear here.
    excluded = {"project_id", "path", "format", "storage_mode", "plugin_name", "completion_answers"}
    assert excluded.isdisjoint(ALL_COPYABLE_FIELDS)


def test_angle_span_is_copied_through_completion_answers():
    source = _session()
    source.config.completion_answers["angle_span"] = 360.0
    target = _session()

    values = copy_settings(source, ["angle_span"])
    paste_settings(target, values)

    assert values == {"angle_span": 360.0}
    assert target.config.completion_answers["angle_span"] == 360.0
    assert "angle_span" in ALL_COPYABLE_FIELDS


def test_angle_span_is_skipped_when_the_source_never_set_one():
    assert copy_settings(_session(), ["angle_span", "center"]) == {"center": None}


def _with_project(session, **sources):
    """Attach a fake imported project whose metadata fields have the given sources."""
    from types import SimpleNamespace

    from core.metadata import MetadataBundle

    meta = MetadataBundle()
    for key, (value, source) in sources.items():
        meta.set(key, value, tier="mandatory_soft", source=source)
    session.run = SimpleNamespace(project=SimpleNamespace(metadata=meta))
    return session


def test_geometry_and_pixel_size_are_copied_when_typed():
    source = _session()
    source.config.completion_answers.update(geometry_type="parallel", voxel_size=166.0, angle_span=360.0)
    target = _session()

    paste_settings(target, copy_settings(source, ["geometry_type", "angle_span", "voxel_size"]))

    assert target.config.completion_answers == {"geometry_type": "parallel", "angle_span": 360.0, "voxel_size": 166.0}


def test_values_loaded_from_the_source_project_are_not_copied():
    source = _session()  # nothing typed: the file supplied them, so there is no answer to copy
    assert copy_settings(source, ["geometry_type", "voxel_size"]) == {}


def test_paste_does_not_override_values_the_target_project_already_loaded():
    target = _with_project(_session(), voxel_size=(20.0, "file"), geometry_type=("cone", "default"))

    paste_settings(target, {"voxel_size": 166.0, "geometry_type": "parallel", "angle_span": 360.0})

    assert "voxel_size" not in target.config.completion_answers      # file-sourced: left alone
    assert target.config.completion_answers["geometry_type"] == "parallel"  # only a default: overwritten
    assert target.config.completion_answers["angle_span"] == 360.0
