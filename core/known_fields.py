"""
This file is the single source of truth for every metadata field the
rest of the software actually depends on -- what it's called, what tier
it belongs to (see core/metadata.py for what tier means), what unit it's
expressed in, and how a missing value for it should eventually get
resolved.

Why this exists, and why it's separate from core/metadata.py:
MetadataBundle and FieldSpec are deliberately generic -- they don't know
what "voxel_size" means, they just track tier/value/source for whatever
key a caller gives them. Before this file existed, tier was a string
literal repeated in every vendor plugin (data_import/vendors/*.py),
which meant nothing outside a plugin's own code enforced that it got a
field's tier right, or even remembered to mention a required field at
all. This file is what plugins and the completion stage both consult
instead of each keeping their own copy of "what does this field mean."

A vendor plugin is still free to report whatever ELSE it wants under
whatever names it wants (see data_import/plugin.py) -- that data just
rides along in the MetadataBundle unexamined, the same way
PathMap.extra carries vendor-specific path info nothing else looks at
(see data_import/path_map.py). This file only defines the fields the
SOFTWARE depends on; it says nothing about what a plugin is allowed to
report beyond that.

To add a new field: add one entry to KNOWN_FIELDS below. Nothing else
needs to change for the field to be picked up by reconcile() /
status() / completion's automatic pass -- see completion/engine.py.

See doc/PROJECT_METADATA.md for the human-readable version of this
table; keep the two in sync.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from core.metadata import MetadataBundle, Source, Tier

# How a missing field's value actually gets filled in, once we know it's
# expected (see reconcile() below for "whether it's expected" at all). A
# field can list more than one strategy, tried in order -- e.g. angles
# is ("derive", "ask"): try to derive it from other known fields first,
# and only fall back to asking the user if that's not possible.
#
#   derive    -- computed automatically from other fields already known
#                (a pure function of the project's current metadata).
#   default   -- a fixed fallback value, applied automatically, always
#                overridable by the user.
#   ask       -- no safe default exists; something outside completion
#                (a notebook cell today, a GUI form later) has to supply it.
#   calculate -- an algorithm proposes a value from the actual image
#                data and a human confirms/adjusts it. This is
#                calibration's job, not completion's -- completion never
#                attempts "calculate" strategy fields.
Strategy = Literal["derive", "default", "ask", "calculate"]


@dataclass(frozen=True)
class FieldDefinition:
    """
    Everything the software needs to know about ONE known metadata key,
    independent of any particular project. Instances of this live only
    in KNOWN_FIELDS below -- a per-project FieldSpec (core/metadata.py)
    is a different, per-instance thing that holds an actual value.
    """

    tier: Tier
    strategies: tuple[Strategy, ...]
    unit: str = ""
    # Only meaningful when "default" is one of the strategies above.
    default: Any = None
    # Only meaningful when this field only matters under some other
    # field's specific value -- e.g. source_to_detector_distance only
    # matters if geometry_type == "cone". Shape: (other_field_key,
    # required_value). None means "always relevant, regardless of any
    # other field".
    condition: Optional[tuple[str, Any]] = None
    description: str = ""


KNOWN_FIELDS: dict[str, FieldDefinition] = {
    # -- Geometry / scan --------------------------------------------------
    "angles": FieldDefinition(
        tier="mandatory_soft",
        strategies=("derive", "ask"),
        unit="degrees",
        description=(
            "Per-projection rotation angle. Derived from angle_span + "
            "projection count + rotation_direction if not given "
            "directly by the vendor."
        ),
    ),
    "geometry_type": FieldDefinition(
        tier="mandatory_soft",
        strategies=("ask",),
        unit="parallel | cone",
        description=(
            "No safe default -- a wrong guess silently corrupts every "
            "downstream calculation. Gates whether the cone-beam-only "
            "fields below become relevant (see their `condition`)."
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
        description="Wrong default only flips orientation, never invalidates the reconstruction.",
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
        description=(
            "Only meaningfully distinct from voxel_size in cone-beam "
            "(magnification); falls back to a plain static default "
            "until a real deriver is registered for it."
        ),
    ),
    # -- Cone-beam-only geometry --------------------------------------------
    "source_to_detector_distance": FieldDefinition(
        tier="mandatory_soft",
        strategies=("ask",),
        unit="mm",
        condition=("geometry_type", "cone"),
        description="Only relevant once geometry_type is known to be cone -- reconcile() won't flag this until then.",
    ),
    "source_to_object_distance": FieldDefinition(
        tier="mandatory_soft",
        strategies=("ask",),
        unit="mm",
        condition=("geometry_type", "cone"),
        description="Same conditional behavior as source_to_detector_distance.",
    ),
    # -- Alignment (computed -- never blocks anything, see core/metadata.py)
    "center_of_rotation": FieldDefinition(
        tier="computed",
        strategies=("calculate",),
        unit="pixels (deviation from image center, sub-pixel)",
        default=0.0,
        description="Resolved by a calibration tool, not completion -- completion never attempts 'calculate' fields.",
    ),
    "tilt_angle": FieldDefinition(
        tier="computed",
        strategies=("calculate",),
        unit="degrees",
        default=0.0,
        description="Same as center_of_rotation: calibration's job.",
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
        description="Pass-through only if a vendor happens to provide it -- never asked for, never affects calculations.",
    ),
}


def reconcile(bundle: MetadataBundle) -> None:
    """
    Make sure every field in KNOWN_FIELDS is represented in `bundle`,
    one way or another. Called right after a plugin's parse_metadata()
    returns (see data_import/importer.py's import_project()).

    A plugin is only required to report fields it actually knows
    something about. This function is what catches the rest: any
    KNOWN_FIELDS key the plugin never touched at all gets added as
    declared-missing, with the tier THIS registry says it should have --
    independent of whether the plugin remembered to. Without this,
    missing_hard()/missing_soft() only ever see what a plugin bothered
    to self-report, which means a plugin that forgets to mention a
    mandatory field fails silently instead of loudly.

    Conditional fields (e.g. source_to_detector_distance, which only
    matters for cone-beam) are only added once their condition is
    actually known to hold -- if geometry_type isn't resolved yet, this
    function leaves them alone rather than flagging a cone-beam-only
    field on what might turn out to be a parallel-beam project. Calling
    this again later (which completion.run() does, via provide()) picks
    them up as soon as their condition becomes true.

    Safe to call more than once -- already-addressed fields are left
    untouched.
    """
    for key, definition in KNOWN_FIELDS.items():
        if bundle.get_field(key) is not None:
            continue  # plugin (or a previous reconcile/completion pass) already addressed this

        if definition.condition is not None:
            dep_key, required_value = definition.condition
            dep_field = bundle.get_field(dep_key)
            if dep_field is None or dep_field.value != required_value:
                continue  # condition not (yet) known to hold -- don't flag this one yet

        bundle.declare_missing(key, tier=definition.tier)


@dataclass
class FieldStatus:
    """One row of what status() below reports for a single known field."""

    key: str
    tier: Tier
    strategies: tuple[Strategy, ...]
    resolved: bool
    value: Any
    source: Source


def status(bundle: MetadataBundle) -> list[FieldStatus]:
    """
    A full readout of every known field's current state: its tier and
    strategy (from this registry) plus whether it's actually resolved
    right now (from the bundle). This is what a notebook -- or later, a
    GUI -- should call to see what a project still needs.

    Calls reconcile() first (harmless if already done), so this always
    reflects every KNOWN_FIELDS entry, not just whatever the plugin
    happened to mention.
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
