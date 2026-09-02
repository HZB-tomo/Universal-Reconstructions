"""
The actual completion logic: run() (the automatic pass) and provide()
(recording an externally-supplied answer). See completion/__init__.py
for why this package doesn't do any asking itself.

register_deriver() is the mechanism for plugging a field's actual
derive logic in, mirroring how data_import/plugin.py's @register_plugin
lets vendor plugins register themselves without importer.py needing to
know about each one by name. The built-in derivers themselves (angles,
angle_span, detector_pixel_size) live in completion/derivers.py, kept
separate from this file the same way vendor plugins are kept out of
data_import/importer.py -- this file is the generic engine, derivers.py
is the field-specific knowledge. Until a deriver is registered for a
given "derive"-strategy field, run() simply can't resolve it
automatically, and falls through to that field's next strategy (if it
has one), or leaves it missing.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from core.known_fields import KNOWN_FIELDS, reconcile
from core.project import ProjectObject

# A deriver takes the project (so it can read any other already-known
# field via project.metadata, or even the array data via DataInterface
# if a real deriver ever needs to) and returns the value it computed, or
# None if it can't yet (missing an input it needs). Keyed by the field
# it derives.
DeriverFn = Callable[[ProjectObject], Optional[Any]]
_DERIVERS: dict[str, DeriverFn] = {}


def register_deriver(key: str):
    """
    Decorator: register a function that computes `key`'s value from
    other fields already on the project. Usage:

        @register_deriver("angles")
        def derive_angles(project: ProjectObject):
            ...  # return None if the inputs it needs aren't known yet

    `key` must be a field in KNOWN_FIELDS whose strategies include
    "derive" (see core/known_fields.py).
    """
    if key not in KNOWN_FIELDS:
        raise ValueError(f"Unknown field '{key}' -- add it to core/known_fields.py first")
    if "derive" not in KNOWN_FIELDS[key].strategies:
        raise ValueError(f"'{key}' isn't a derivable field (see its strategies in core/known_fields.py)")

    def wrapper(fn: DeriverFn) -> DeriverFn:
        _DERIVERS[key] = fn
        return fn

    return wrapper


def run(project: ProjectObject) -> None:
    """
    The automatic pass: for every field that's still missing, try its
    strategies in order, stopping at the first one that actually
    resolves it. "ask" and "calculate" are never attempted here -- they
    require a human, which is exactly what run() doesn't do (see module
    docstring).

    Repeats until a full pass makes no further progress, rather than
    just running once -- derivers can depend on each other (angles
    needs rotation_direction, which only gets its default applied
    during the same run()), and KNOWN_FIELDS' dict order doesn't
    guarantee dependencies are already resolved by the time a
    dependent field's turn comes up in a single pass. Bounded by
    len(KNOWN_FIELDS) extra passes, which is more than enough for any
    dependency chain that doesn't involve a cycle -- a cycle just means
    no progress is made and the loop exits normally, leaving the
    involved fields missing.

    Safe to call repeatedly across separate calls too: an
    already-resolved field is left untouched, so calling this again
    after provide() supplies one new answer just tries to make further
    progress with it (e.g. a fresh conditional field that reconcile()
    only just flagged as relevant).
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
                # "ask" / "calculate": nothing automatic to do here -- leave it missing.

        if not progress:
            break


def provide(project: ProjectObject, key: str, value: Any) -> None:
    """
    Record a value for `key` that something outside this package
    collected from a user (a notebook cell today; a GUI form later),
    then re-run the automatic pass. One new answer can be exactly what
    unblocks a "derive" field elsewhere (e.g. supplying angle_span can
    now let angles derive), and can also newly satisfy a conditional
    field's condition (e.g. answering geometry_type="cone" makes
    source_to_detector_distance become relevant -- see
    core/known_fields.reconcile()).
    """
    if key not in KNOWN_FIELDS:
        raise ValueError(f"Unknown field '{key}' -- add it to core/known_fields.py first")

    project.metadata.set(key, value, tier=KNOWN_FIELDS[key].tier, source="user")
    run(project)
