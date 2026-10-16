"""
Copying ProjectConfig/ProjectSession settings from one project to
another -- ProjectPanel's right-click "Copy settings"/"Copy selected
settings..."/"Paste settings".

COPYABLE_SETTINGS is grouped by category, matching how
CopySettingsDialog presents them -- one QGroupBox per category. Only
settings that don't come from the project's own file (geometry_type,
voxel_size, path, format, ...) are here; those are excluded on
purpose, since pasting them onto a different project's data would be
wrong more often than not. geometry_type, angle_span and voxel_size
(pixel size) are the exception, but only for values the user typed in:
a value a project's own file supplies is never copied from it, nor
pasted over on a project that has its own (see _loaded_from_project()).
Projects from one acquisition usually share these.

Every entry is (field_name, label). field_name is a ProjectConfig
attribute name, except "crop_yx", special-cased to
ProjectSession.export_crop_yx (GUI-only state, not a config field --
see gui/session.py), and the _ANSWER_FIELDS, special-cased to
config.completion_answers (where the GUI stores what the user types).
"""
from __future__ import annotations

from typing import Any

from core.data_interface import DataInterface
from gui.session import ProjectSession

_ANSWER_FIELDS = ("geometry_type", "angle_span", "voxel_size")

COPYABLE_SETTINGS: dict[str, list[tuple[str, str]]] = {
    "Scan": [
        ("geometry_type", "Geometry type"),
        ("angle_span", "Angle span"),
        ("voxel_size", "Pixel size"),
    ],
    "Calibration": [
        ("center", "Center of rotation"),
        ("tilt_deg", "Tilt"),
    ],
    "Reconstruction": [
        ("rotate_deg", "Rotate"),
        ("reconstruction_backend", "Algorithm"),
        ("device", "Device"),
        ("reconstruction_options", "Algorithm options (e.g. iterations)"),
    ],
    "Refinements": [
        ("projections_normalized", "Projections already flat/dark corrected"),
        ("ring_correction", "Ring correction (on/off)"),
        ("ring_method", "Ring method"),
        ("ring_options", "Ring options"),
        ("zinger_removal", "Zinger removal (on/off)"),
        ("zinger_threshold", "Zinger threshold"),
        ("zinger_size", "Zinger size"),
    ],
    "Postprocess": [
        ("dtype", "Output dtype"),
        ("dtype_clip_min", "Clip min"),
        ("dtype_clip_max", "Clip max"),
    ],
    "Crop": [
        ("crop_yx", "Crop rectangle"),
    ],
    "Save settings": [
        ("export_format", "Export format"),
        ("stream", "Stream to disk"),
    ],
}

ALL_COPYABLE_FIELDS = [name for fields in COPYABLE_SETTINGS.values() for name, _ in fields]


def _read(session: ProjectSession, field_name: str) -> Any:
    if field_name == "crop_yx":
        return session.export_crop_yx
    if field_name in _ANSWER_FIELDS:
        return session.config.completion_answers.get(field_name)
    return getattr(session.config, field_name)


def _write(session: ProjectSession, field_name: str, value: Any) -> None:
    if field_name == "crop_yx":
        session.export_crop_yx = value
    elif field_name in _ANSWER_FIELDS:
        session.config.completion_answers[field_name] = value
    else:
        setattr(session.config, field_name, value)


def _loaded_from_project(session: ProjectSession, field_name: str) -> bool:
    """True if `field_name`'s current value on `session` was read from the
    project's own file (only knowable once the project is imported)."""
    project = getattr(session.run, "project", None)
    return project is not None and DataInterface(project).get_metadata_source(field_name) == "file"


def copy_settings(session: ProjectSession, field_names: list[str]) -> dict[str, Any]:
    """A plain {field_name: value} dict -- dict-valued fields
    (reconstruction_options/ring_options) are copied, not aliased, so
    the clipboard can't be mutated by later edits to `session`."""
    values = {}
    for name in field_names:
        value = _read(session, name)
        if name in _ANSWER_FIELDS and value is None:
            continue  # never typed (the file supplied it, or none yet): nothing to copy
        values[name] = dict(value) if isinstance(value, dict) else value
    return values


def paste_settings(session: ProjectSession, values: dict[str, Any]) -> None:
    """Writes every entry in `values` onto `session`. Dict-valued
    entries are copied, not aliased, so two pasted-onto sessions never
    end up sharing the same mutable dict."""
    for name, value in values.items():
        if name in _ANSWER_FIELDS and _loaded_from_project(session, name):
            continue  # this project's own file already supplies it
        _write(session, name, dict(value) if isinstance(value, dict) else value)
