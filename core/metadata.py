"""
This file defines how we store a single piece of metadata (a "field",
like voxel size or center of rotation) and a collection of them (a
"bundle"). Everything here is about tracking not just VALUES but also
HOW IMPORTANT each value is and WHERE it came from.

Why this exists: not all metadata is equally critical. If "projections"
is missing, the software literally cannot function. If "voxel size" is
missing, the software can still load the data but needs to ask the user
for it before running certain steps (like reconstruction). If "operator
name" is missing, nobody cares. And some values (like center of rotation)
aren't really "missing or present" at all -- they might arrive as a
rough guess from the file, then get overwritten by a computed value
later in the pipeline.

We capture all of that with a "tier" per field, so the rest of the
software can ask simple questions like "is this project usable yet?"
or "does the user need to be prompted?" without every module having to
re-invent that logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

# The four tiers, as agreed:
#   mandatory_hard -> if missing, importing must fail outright (e.g. projections)
#   mandatory_soft -> if missing, importing still succeeds, but the project
#                      is flagged as "needs user input" before later stages can run
#   optional       -> nice to have, no consequence if missing
#   computed       -> usually filled in later by an algorithm (e.g. center of
#                      rotation found by an auto-alignment step). May also
#                      arrive from the file as a starting guess.
Tier = Literal["mandatory_hard", "mandatory_soft", "optional", "computed"]

# Where a field's current value came from. Useful later for questions like
# "should I trust this or recompute it?" without needing a full history log
# (we deliberately do NOT keep value history -- overwriting is fine, per
# the earlier discussion; we just remember the most recent source).
#   file     -- read straight out of the vendor's file by a plugin.
#   user     -- typed in by a person (a notebook cell today, a GUI later).
#   computed -- written by an algorithm, e.g. a calibration tool's estimate.
#   default  -- the completion stage's automatic fallback value (see
#               core/known_fields.py) -- distinct from "user" so it stays
#               visible that nobody has actually confirmed this value yet.
#   derived  -- computed automatically by completion from other fields
#               already known (see core/known_fields.py), as opposed to
#               "computed", which implies an algorithm ran over image data.
Source = Literal["file", "user", "computed", "default", "derived", None]


@dataclass
class FieldSpec:
    """
    One single metadata field, e.g. "voxel_size" or "center_of_rotation".

    This is intentionally a plain data holder (a dataclass) with no
    behaviour of its own -- all the logic for interpreting fields lives
    in MetadataBundle below, so FieldSpec stays trivial to read, copy,
    and serialize (e.g. to JSON for saving a project file later).
    """

    key: str          # the field's name, e.g. "voxel_size"
    tier: Tier         # which of the four tiers this field belongs to
    value: Any = None  # the actual value, or None if not yet known
    source: Source = None  # where the current value came from

    @property
    def is_missing(self) -> bool:
        """True if this field has no value yet. Used by MetadataBundle
        to figure out which mandatory fields still need attention."""
        return self.value is None


class MetadataBundle:
    """
    A collection of FieldSpecs for one project, plus convenience methods
    for the two questions the rest of the software actually needs to ask:

      1. "Can we even start working with this data?" -> missing_hard()
      2. "Should we ask the user for anything before continuing?" -> missing_soft()

    A vendor plugin builds one of these while reading a file (see
    data_import/plugin.py), and the resulting ProjectObject carries it
    around for the rest of the pipeline to consult.
    """

    def __init__(self):
        # Plain dict keyed by field name -- simplest possible storage,
        # no need for anything fancier at this stage.
        self._fields: dict[str, FieldSpec] = {}

    def set(self, key: str, value: Any, tier: Tier, source: Source = "file"):
        """
        Record a field WITH a known value. Typically called by a vendor
        plugin right after it successfully read something from the file
        (source="file"), but can also be used for user-provided values
        (source="user").
        """
        self._fields[key] = FieldSpec(key=key, tier=tier, value=value, source=source)

    def declare_missing(self, key: str, tier: Tier):
        """
        Record that a field is EXPECTED but was NOT found. This is
        different from simply not calling set() at all: declaring it
        missing means the field shows up in missing_hard()/missing_soft()
        so the importer/UI knows to complain about it or prompt for it.
        If you never declare a field at all, the software has no way of
        knowing it was ever expected.
        """
        self._fields[key] = FieldSpec(key=key, tier=tier, value=None, source=None)

    def get(self, key: str, default: Any = None) -> Any:
        """Read a field's current value, or `default` if it isn't set."""
        field = self._fields.get(key)
        return field.value if field is not None else default

    def get_field(self, key: str) -> Optional[FieldSpec]:
        """Read the full FieldSpec (value + tier + source), not just the value.
        Useful when you need to know provenance, e.g. before overwriting
        a computed value you might want to check `source == "file"` first."""
        return self._fields.get(key)

    def set_computed(self, key: str, value: Any):
        """
        Overwrite a field with a freshly computed value (e.g. an
        auto-alignment algorithm just calculated center_of_rotation).
        Deliberately just overwrites -- no history is kept, per our
        earlier decision to keep this simple. If the field previously
        had a "file"-sourced starting guess, that guess is now gone;
        only the latest value matters.
        """
        self._fields[key] = FieldSpec(key=key, tier="computed", value=value, source="computed")

    def missing_hard(self) -> list[str]:
        """Names of all mandatory_hard fields that have no value.
        A non-empty result means: importing should not be considered
        successful. See ProjectObject.is_ready_for_import()."""
        return [k for k, f in self._fields.items() if f.tier == "mandatory_hard" and f.is_missing]

    def missing_soft(self) -> list[str]:
        """Names of all mandatory_soft fields that have no value.
        A non-empty result means: the project loaded fine, but the user
        should be prompted for these before certain pipeline stages can
        run. See ProjectObject.needs_user_input()."""
        return [k for k, f in self._fields.items() if f.tier == "mandatory_soft" and f.is_missing]

    def all_fields(self) -> dict[str, FieldSpec]:
        """Everything at once -- mainly useful for debugging, logging,
        or building a UI that lists every known field."""
        return dict(self._fields)

    def __repr__(self):
        return f"MetadataBundle({list(self._fields.keys())})"
