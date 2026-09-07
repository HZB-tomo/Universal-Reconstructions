"""
Defines a single piece of metadata (a "field", e.g. voxel size or
center of rotation) and a collection of them (a "bundle"). Tracks each
value's tier (how critical it is) and source (where it came from).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

# mandatory_hard: if missing, importing fails outright (e.g. projections).
# mandatory_soft: if missing, importing still succeeds, but the project
#                 is flagged as needing user input before later stages run.
# optional:       no consequence if missing.
# computed:       filled in later by an algorithm (e.g. center of rotation
#                 found by an auto-alignment step). May start as a guess
#                 from the file.
Tier = Literal["mandatory_hard", "mandatory_soft", "optional", "computed"]

# Where a field's current value came from. No value history is kept;
# only the most recent source is tracked.
#   file:     read from the vendor's file by a plugin.
#   user:     entered by a person.
#   computed: written by an algorithm.
#   default:  the completion stage's automatic fallback value (see
#             core/known_fields.py), distinct from "user" so it stays
#             visible that nobody has confirmed this value.
#   derived:  computed automatically by completion from other known
#             fields, as opposed to "computed", which implies an
#             algorithm ran over image data.
Source = Literal["file", "user", "computed", "default", "derived", None]


@dataclass
class FieldSpec:
    """
    One metadata field, e.g. "voxel_size" or "center_of_rotation". A
    plain data holder; logic for interpreting fields lives in
    MetadataBundle.
    """

    key: str
    tier: Tier
    value: Any = None
    source: Source = None

    @property
    def is_missing(self) -> bool:
        """True if this field has no value yet."""
        return self.value is None


class MetadataBundle:
    """
    A collection of FieldSpecs for one project, with methods for:

      1. Can this data be used at all? -> missing_hard()
      2. Should the user be asked for anything? -> missing_soft()

    A vendor plugin builds one of these while reading a file (see
    data_import/plugin.py); the resulting ProjectObject carries it for
    the rest of the pipeline.
    """

    def __init__(self):
        self._fields: dict[str, FieldSpec] = {}

    def set(self, key: str, value: Any, tier: Tier, source: Source = "file"):
        """Record a field with a known value."""
        self._fields[key] = FieldSpec(key=key, tier=tier, value=value, source=source)

    def declare_missing(self, key: str, tier: Tier):
        """
        Record that a field is expected but was not found. Distinct
        from never calling set(): a declared-missing field shows up in
        missing_hard()/missing_soft(), so the importer or UI knows to
        prompt for it.
        """
        self._fields[key] = FieldSpec(key=key, tier=tier, value=None, source=None)

    def get(self, key: str, default: Any = None) -> Any:
        """Read a field's current value, or `default` if unset."""
        field = self._fields.get(key)
        return field.value if field is not None else default

    def get_field(self, key: str) -> Optional[FieldSpec]:
        """Read the full FieldSpec (value, tier, source)."""
        return self._fields.get(key)

    def set_computed(self, key: str, value: Any):
        """
        Overwrite a field with a freshly computed value (e.g. an
        auto-alignment algorithm's result for center_of_rotation). No
        history is kept; a prior value is replaced.
        """
        self._fields[key] = FieldSpec(key=key, tier="computed", value=value, source="computed")

    def missing_hard(self) -> list[str]:
        """Names of mandatory_hard fields with no value. Non-empty means
        importing should not be considered successful. See
        ProjectObject.is_ready_for_import()."""
        return [k for k, f in self._fields.items() if f.tier == "mandatory_hard" and f.is_missing]

    def missing_soft(self) -> list[str]:
        """Names of mandatory_soft fields with no value. Non-empty means
        the user should be prompted before certain pipeline stages can
        run. See ProjectObject.needs_user_input()."""
        return [k for k, f in self._fields.items() if f.tier == "mandatory_soft" and f.is_missing]

    def all_fields(self) -> dict[str, FieldSpec]:
        """Every field at once."""
        return dict(self._fields)

    def __repr__(self):
        return f"MetadataBundle({list(self._fields.keys())})"
