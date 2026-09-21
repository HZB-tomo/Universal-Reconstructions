"""
Copying ProjectConfig/ProjectSession settings from one project to
another -- ProjectPanel's right-click "Copy settings"/"Copy selected
settings..."/"Paste settings".

COPYABLE_SETTINGS is grouped by category, matching how
CopySettingsDialog presents them -- one QGroupBox per category. Only
settings that don't come from the project's own file (geometry_type,
angle_span, voxel_size, path, format, ...) are here; those are
excluded on purpose, since pasting them onto a different project's
data would be wrong more often than not.

Every entry is (field_name, label). field_name is a ProjectConfig
attribute name, except "crop_yx", special-cased to
ProjectSession.export_crop_yx (GUI-only state, not a config field --
see gui/session.py).
"""
from __future__ import annotations

from typing import Any

from gui.session import ProjectSession

COPYABLE_SETTINGS: dict[str, list[tuple[str, str]]] = {
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
    return getattr(session.config, field_name)


def _write(session: ProjectSession, field_name: str, value: Any) -> None:
    if field_name == "crop_yx":
        session.export_crop_yx = value
    else:
        setattr(session.config, field_name, value)


def copy_settings(session: ProjectSession, field_names: list[str]) -> dict[str, Any]:
    """A plain {field_name: value} dict -- dict-valued fields
    (reconstruction_options/ring_options) are copied, not aliased, so
    the clipboard can't be mutated by later edits to `session`."""
    values = {}
    for name in field_names:
        value = _read(session, name)
        values[name] = dict(value) if isinstance(value, dict) else value
    return values


def paste_settings(session: ProjectSession, values: dict[str, Any]) -> None:
    """Writes every entry in `values` onto `session`. Dict-valued
    entries are copied, not aliased, so two pasted-onto sessions never
    end up sharing the same mutable dict."""
    for name, value in values.items():
        _write(session, name, dict(value) if isinstance(value, dict) else value)
