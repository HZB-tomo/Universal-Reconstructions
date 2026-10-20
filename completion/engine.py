"""
The completion logic: run() (the automatic pass) and provide()
(recording an externally-supplied answer).

register_deriver() registers a field's derive logic. The built-in
derivers (angles, angle_span, detector_pixel_size) live in
completion/derivers.py. Until a deriver is registered for a given
"derive"-strategy field, run() falls through to that field's next
strategy, or leaves it missing.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from core.known_fields import KNOWN_FIELDS, reconcile
from core.project import ProjectObject

# A deriver takes the project and returns the value it computed, or
# None if it can't yet. Keyed by the field it derives.
DeriverFn = Callable[[ProjectObject], Optional[Any]]
_DERIVERS: dict[str, DeriverFn] = {}


def register_deriver(key: str):
    """
    Decorator: register a function that computes `key`'s value from
    other fields already on the project.

        @register_deriver("angles")
        def derive_angles(project: ProjectObject):
            ...  # return None if the inputs it needs aren't known yet

    `key` must be a field in KNOWN_FIELDS whose strategies include
    "derive".
    """
    if key not in KNOWN_FIELDS:
        raise ValueError(f"Unknown field '{key}'. Add it to core/known_fields.py first")
    if "derive" not in KNOWN_FIELDS[key].strategies:
        raise ValueError(f"'{key}' isn't a derivable field (see its strategies in core/known_fields.py)")

    def wrapper(fn: DeriverFn) -> DeriverFn:
        _DERIVERS[key] = fn
        return fn

    return wrapper


def run(project: ProjectObject) -> None:
    """
    The automatic pass: for every field still missing, try its
    strategies in order, stopping at the first that resolves it. "ask"
    and "calculate" are never attempted here.

    Repeats until a full pass makes no further progress: derivers can
    depend on each other (angles needs rotation_direction, which gets
    its default applied during the same run()), and KNOWN_FIELDS' dict
    order does not guarantee a dependency is already resolved by the
    time a dependent field's turn comes up in a single pass. Bounded by
    len(KNOWN_FIELDS) extra passes; a cycle simply makes no progress and
    the loop exits, leaving the involved fields missing.

    Safe to call repeatedly. An already-resolved field is left
    untouched.
    """
    reconcile(project.metadata)

    for _ in range(len(KNOWN_FIELDS) + 1):
        progress = False

        for key, definition in KNOWN_FIELDS.items():
            field_spec = project.metadata.get_field(key)
            if field_spec is None or not field_spec.is_missing:
                continue

            for strategy in definition.strategies:
                if strategy == "derive":
                    deriver = _DERIVERS.get(key)
                    if deriver is None:
                        continue
                    value = deriver(project)
                    if value is None:
                        continue
                    project.metadata.set(key, value, tier=definition.tier, source="derived")
                    progress = True
                    break
                elif strategy == "default":
                    if definition.default is None:
                        continue
                    project.metadata.set(key, definition.default, tier=definition.tier, source="default")
                    progress = True
                    break
                # "ask" / "calculate": nothing automatic to do; leave it missing.

        if not progress:
            break


def provide(project: ProjectObject, key: str, value: Any) -> None:
    """
    Record a value for `key` collected from a user, then re-run the
    automatic pass. One new answer can unblock a "derive" field
    elsewhere (e.g. supplying angle_span lets angles derive), or
    satisfy a conditional field's condition (e.g. geometry_type="cone"
    makes source_to_detector_distance relevant; see
    core/known_fields.reconcile()).
    """
    if key not in KNOWN_FIELDS:
        raise ValueError(f"Unknown field '{key}'. Add it to core/known_fields.py first")

    ## this is for when we get angles  from metadata, but we still keep it changeable 

    if key in {"angle_span", "rotation_direction"}:
        angles = project.metadata.get_field("angles")
        if angles is not None and angles.source == "derived":
            project.metadata.declare_missing("angles", tier=KNOWN_FIELDS["angles"].tier)

    project.metadata.set(key, value, tier=KNOWN_FIELDS[key].tier, source="user")
    run(project)
