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
