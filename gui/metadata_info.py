"""
What the metadata toolbox shows: every metadata field of one project, grouped by
tier, with its value, where the value came from, and what is missing. Qt-free.

Built from the project's MetadataBundle (what the import plugin read and the
completion stage added) and core/known_fields.py (what the software depends on,
including fields the file did not mention at all).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from core.known_fields import KNOWN_FIELDS
from core.metadata import MetadataBundle
from gui.session import ProjectSession

# Display order and wording of the tiers (core/metadata.py).
GROUPS = (
    ("mandatory_hard", "Must be in the file"),
    ("mandatory_soft", "Needed (can be filled in)"),
    ("optional", "Optional"),
    ("computed", "Calculated by the software"),
    ("other", "Other, from the file"),
)

STATUS_TEXT = {
    "read": "read from file",
    "user": "set by you",
    "default": "default",
    "derived": "derived",
    "computed": "calculated",
    "settings": "set in settings",
    "missing": "MISSING",
    "later": "not yet calculated",
    "not_needed": "not needed",
}
HOW_TO_FIX = {
    "derive": "can be derived from other fields",
    "default": "has a default",
    "ask": "you can enter it (Basic settings, or the completion answers)",
    "calculate": "the software calculates it (Find center / Find tilt)",
}
PROBLEM_STATUSES = ("missing",)


@dataclass
class MetadataRow:
    key: str
    group: str                 # one of GROUPS' keys
    status: str                # one of STATUS_TEXT's keys
    value: str = ""
    unit: str = ""
    source: str = ""
    note: str = ""             # description, or how a missing value gets resolved


@dataclass
class MetadataView:
    title: str
    rows: list = field(default_factory=list)
    message: str = ""          # shown instead of rows when there is nothing to list (not imported yet, ...)
    notice: str = ""           # a problem to point out above the rows (completion could not fill in everything)

    def group(self, name: str) -> list:
        return [r for r in self.rows if r.group == name]

    @property
    def counts(self) -> dict:
        out: dict = {}
        for row in self.rows:
            out[row.status] = out.get(row.status, 0) + 1
        return out

    @property
    def summary(self) -> str:
        if self.message:
            return self.message
        counts = self.counts
        missing = counts.get("missing", 0)
        read = counts.get("read", 0)
        parts = [f"{read} read from the file"]
        for status in ("user", "default", "derived", "computed", "settings"):
            if counts.get(status):
                parts.append(f"{counts[status]} {STATUS_TEXT[status]}")
        parts.append(f"{missing} missing" if missing else "nothing missing")
        text = ", ".join(parts)
        return f"{text}. {self.notice}" if self.notice else text


def format_value(value: Any, limit: int = 80) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray) or (isinstance(value, (list, tuple)) and len(value) > 6):
        array = np.asarray(value)
        if array.dtype.kind in "fiu" and array.size:
            return f"array {array.shape}, {array.min():g} .. {array.max():g}"
        return f"array {array.shape}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):g}"
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _status_of(source: Optional[str]) -> str:
    return {"file": "read", "user": "user", "default": "default", "derived": "derived", "computed": "computed"}.get(source or "", "read")


def _applies(bundle: MetadataBundle, definition) -> bool:
    """Whether a conditional field matters for this project (cone-beam distances only for cone beam)."""
    if definition.condition is None:
        return True
    other, required = definition.condition
    spec = bundle.get_field(other)
    return spec is not None and spec.value == required


def metadata_view(session: Optional[ProjectSession]) -> MetadataView:
    if session is None:
        return MetadataView("", message="Select a project to see its metadata.")
    title = session.project_id
    if not session.imported:
        why = f"failed to import: {session.error}" if session.error else "is still importing"
        return MetadataView(title, message=f"{session.project_id} {why}.")
    bundle: MetadataBundle = session.run.project.metadata
    config = session.config
    rows: list = []
    seen: set = set()

    for key, definition in KNOWN_FIELDS.items():
        seen.add(key)
        spec = bundle.get_field(key)
        note = definition.description
        fix = ", ".join(HOW_TO_FIX[s] for s in definition.strategies)
        if spec is not None and not spec.is_missing:
            rows.append(MetadataRow(
                key, definition.tier, _status_of(spec.source), format_value(spec.value), definition.unit,
                spec.source or "", note,
            ))
            continue
        if not _applies(bundle, definition):
            rows.append(MetadataRow(key, definition.tier, "not_needed", "", definition.unit, "",
                                    f"only for {definition.condition[1]}-beam projects"))
            continue
        override = {"center_of_rotation": config.center, "tilt_angle": config.tilt_deg}.get(key)
        if override is not None:
            rows.append(MetadataRow(key, definition.tier, "settings", format_value(override), definition.unit, "settings", note))
        elif definition.tier == "computed":
            rows.append(MetadataRow(key, definition.tier, "later", "", definition.unit, "", fix or note))
        else:
            rows.append(MetadataRow(key, definition.tier, "missing", "", definition.unit, "", fix or note))

    for key, spec in sorted(bundle.all_fields().items()):
        if key in seen:
            continue
        group = "other" if key.startswith("extra.") else spec.tier
        if spec.is_missing:
            rows.append(MetadataRow(key, group, "missing", "", "", "", "expected by the plugin, not found"))
        else:
            rows.append(MetadataRow(key, group, _status_of(spec.source), format_value(spec.value), "", spec.source or "", ""))

    order = {name: i for i, (name, _label) in enumerate(GROUPS)}
    rows.sort(key=lambda r: order.get(r.group, len(order)))      # stable: keeps the catalog order within a tier
    return MetadataView(title, rows, notice=session.error or "")
