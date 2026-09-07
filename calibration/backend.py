"""
The calibration plugin contract (CalibrationBackend) and the registry
backends sign up with -- the same idiom as reconstruction/backend.py,
for the same reason: explicit selection, no signature/detection needed,
because the caller already knows which library it wants.

Deliberately the same shape as reconstruction/backend.py for a second
reason too: if algotom (the one backend implemented today) is ever
unavailable or removed, a from-scratch reimplementation of its
published algorithm only has to satisfy this one small contract to
slot in as a replacement -- nothing else in this package, or in
calibration/api.py, needs to change.

Exactly one method: find one 180-degree sinogram's center of rotation.
Everything else calibration does -- axis-lean/tilt detection via
find_tilt_and_center() -- is orchestration built on top of this single
primitive (see api.py), not something a backend needs to know about.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class CalibrationBackend(ABC):
    """
    Base class every calibration plugin subclasses. Exactly one method
    to implement: find_center(). Whatever algorithm/library it uses
    internally (Vo's published method via algotom today; a from-scratch
    reimplementation of the same published algorithm if algotom is ever
    unavailable; a different published method entirely) is this
    method's own business -- see backends/algotom_cor.py for the one
    built-in implementation and its citation.

    Class attribute to set on your subclass:
        name: a short unique string, e.g. "algotom". Used for lookup via get_backend().
    """

    name: str

    @abstractmethod
    def find_center(
        self,
        sino_180: np.ndarray,
        start: Optional[float] = None,
        stop: Optional[float] = None,
        **options,
    ) -> float:
        """
        `sino_180` is a single 2D sinogram, (n_angles, width), covering
        a 0-180 degree angular range -- the same parallel-beam
        convention every center-finding method in this project uses.

        `start`/`stop` bound the search range, in the same column-index
        units reconstruct()'s own `center` parameter expects. Always
        pass an explicit, appropriately wide range rather than relying
        on a backend's own default -- verified directly against
        algotom's own default (`width/2 +/- width/16`): it's too narrow
        for anything beyond a small COR deviation. calibration/api.py's
        find_center() always supplies an explicit range for this reason.

        Returns the estimated center of rotation, in the same units.
        """


_REGISTRY: dict[str, CalibrationBackend] = {}


def register_backend(cls):
    """
    Decorator: put @register_backend above a CalibrationBackend
    subclass and it's automatically instantiated and added to the
    registry as soon as its module is imported -- same pattern as
    reconstruction/backend.py's @register_backend.
    """
    instance = cls()
    _REGISTRY[instance.name] = instance
    return cls


def get_backend(name: str) -> CalibrationBackend:
    """Look up a registered backend by name. Raises KeyError, listing
    what IS registered, if `name` isn't found -- selection here is
    always explicit, never auto-detected."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown calibration backend '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def clear_registry():
    """Empty the registry. Exists for tests, mirroring reconstruction.backend.clear_registry()."""
    _REGISTRY.clear()
