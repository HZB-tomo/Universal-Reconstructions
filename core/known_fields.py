"""
The catalog of every metadata field the software depends on: its tier
(see core/metadata.py), unit, and how a missing value is resolved.

MetadataBundle and FieldSpec are generic; they do not know what
"voxel_size" means. This file is what plugins and the completion stage
both consult for field meaning, instead of each keeping its own copy.

A vendor plugin may still report other fields under other names (see
data_import/plugin.py); that data rides along in the MetadataBundle
unexamined. This file defines only the fields the software depends on.

To add a field: add an entry to KNOWN_FIELDS. reconcile() / status() /
completion's automatic pass pick it up automatically (see
completion/engine.py).

See doc/PROJECT_METADATA.md for the human-readable version; keep the
two in sync.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from core.metadata import MetadataBundle, Source, Tier

# How a missing field's value gets filled in, once it is known to be
# expected (see reconcile()). A field can list more than one strategy,
# tried in order; angles is ("derive", "ask").
#
#   derive:    computed automatically from other known fields.
#   default:   a fixed fallback value, applied automatically, always
#              overridable by the user.
#   ask:       no safe default; something outside completion supplies it.
#   calculate: an algorithm proposes a value from the image data and a
#              human confirms or adjusts it. This is calibration's job;
#              completion never attempts "calculate" fields.
Strategy = Literal["derive", "default", "ask", "calculate"]


@dataclass(frozen=True)
class FieldDefinition:
    """
    What the software needs to know about one known metadata key,
    independent of any project. Instances live only in KNOWN_FIELDS. A
    per-project FieldSpec (core/metadata.py) holds an actual value.
    """

    tier: Tier
    strategies: tuple[Strategy, ...]
    unit: str = ""
    default: Any = None  # meaningful only when "default" is a strategy
    # Meaningful only when this field matters conditionally on another
    # field's value, e.g. source_to_detector_distance only when
    # geometry_type == "cone". Shape: (other_field_key, required_value).
    # None means always relevant.
    condition: Optional[tuple[str, Any]] = None
    description: str = ""


KNOWN_FIELDS: dict[str, FieldDefinition] = {
    # -- Geometry / scan --------------------------------------------------
    "angles": FieldDefinition(
        tier="mandatory_soft",
        strategies=("derive", "ask"),
        unit="degrees",
        description=(
            "Per-projection rotation angle. Derived from angle_span, "
            "projection count, and rotation_direction if not given "
            "directly by the vendor."
        ),
    ),
    "geometry_type": FieldDefinition(
        tier="mandatory_soft",
        strategies=("ask",),
        unit="parallel | cone",
        description=(
            "No safe default. Gates whether the cone-beam-only fields "
            "below become relevant (see their `condition`)."
        ),
    ),
    "angle_span": FieldDefinition(
        tier="mandatory_soft",
        strategies=("derive", "ask"),
        unit="degrees",
        description="Total angular coverage. Derived from angles (max - min) when angles are already known.",
    ),
    "rotation_direction": FieldDefinition(
        tier="optional",
        strategies=("default",),
        unit="cw | ccw",
        default="ccw",
        description="A wrong default flips orientation; it does not invalidate the reconstruction.",
    ),
    # -- Pixel / detector --------------------------------------------------
    "voxel_size": FieldDefinition(
        tier="mandatory_soft",
        strategies=("default",),
        unit="um",
        default=1.0,
    ),
    "detector_pixel_size": FieldDefinition(
        tier="mandatory_soft",
        strategies=("derive", "default"),
        unit="um",
        default=1.0,
        description="Distinct from voxel_size only in cone-beam (magnification).",
    ),
    # -- Cone-beam-only geometry --------------------------------------------
    "source_to_detector_distance": FieldDefinition(
        tier="mandatory_soft",
        strategies=("ask",),
        unit="mm",
        condition=("geometry_type", "cone"),
        description="Relevant only once geometry_type is known to be cone.",
    ),
    "source_to_object_distance": FieldDefinition(
        tier="mandatory_soft",
        strategies=("ask",),
        unit="mm",
        condition=("geometry_type", "cone"),
        description="Same conditional behavior as source_to_detector_distance.",
    ),
    # -- Alignment (computed; never blocks anything, see core/metadata.py)
    "center_of_rotation": FieldDefinition(
        tier="computed",
        strategies=("calculate",),
        unit="pixels (deviation from image center, sub-pixel)",
        default=0.0,
        description="Resolved by calibration, not completion.",
    ),
    "tilt_angle": FieldDefinition(
        tier="computed",
        strategies=("calculate",),
        unit="degrees",
        default=0.0,
        description="Resolved by calibration, not completion.",
    ),
    # -- Provenance ----------------------------------------------------------
    "operator": FieldDefinition(tier="optional", strategies=("ask",), unit="text"),
    "instrument_name": FieldDefinition(tier="optional", strategies=(), unit="text"),
    "acquisition_datetime": FieldDefinition(tier="optional", strategies=(), unit="timestamp"),
    "sample_name": FieldDefinition(tier="optional", strategies=("ask",), unit="text"),
    "energy": FieldDefinition(
        tier="optional",
        strategies=(),
        unit="keV",
        description="Pass-through only if a vendor provides it.",
    ),
}


def reconcile(bundle: MetadataBundle) -> None:
    """
    Ensure every field in KNOWN_FIELDS is represented in `bundle`.
    Called after a plugin's parse_metadata() returns (see
    data_import/importer.py's import_project()).

    Any KNOWN_FIELDS key the plugin did not report is added as
    declared-missing, with the tier this registry specifies.

    Conditional fields (e.g. source_to_detector_distance, cone-beam
    only) are added only once their condition is known to hold. Calling
    this again later (completion.run(), via provide()) picks them up as
    soon as their condition becomes true.

    Safe to call more than once; already-addressed fields are left
    untouched.
    """
    for key, definition in KNOWN_FIELDS.items():
        if bundle.get_field(key) is not None:
            continue

        if definition.condition is not None:
            dep_key, required_value = definition.condition
            dep_field = bundle.get_field(dep_key)
            if dep_field is None or dep_field.value != required_value:
                continue

        bundle.declare_missing(key, tier=definition.tier)


@dataclass
class FieldStatus:
    """One row of what status() reports for a single known field."""

    key: str
    tier: Tier
    strategies: tuple[Strategy, ...]
    resolved: bool
    value: Any
    source: Source


def status(bundle: MetadataBundle) -> list[FieldStatus]:
    """
    A readout of every known field's current state: tier and strategy
    from this registry, plus whether it is resolved in `bundle`.

    Calls reconcile() first.
    """
    reconcile(bundle)
    result = []
    for key, definition in KNOWN_FIELDS.items():
        field_spec = bundle.get_field(key)
        result.append(
            FieldStatus(
                key=key,
                tier=definition.tier,
                strategies=definition.strategies,
                resolved=field_spec is not None and not field_spec.is_missing,
                value=field_spec.value if field_spec is not None else None,
                source=field_spec.source if field_spec is not None else None,
            )
        )
    return result
