"""
CalibrationBackend: the plugin contract for center-of-rotation
backends, and the registry backends register with.

Backend selection is always explicit: a name looked up via
get_backend(), never auto-detected.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class CalibrationBackend(ABC):
    """
    Base class for a calibration backend. Subclasses implement exactly
    one method: find_center().

    Class attribute to set on a subclass:
        name: a short unique string, e.g. "algotom". Used for lookup
            via get_backend().
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
        `sino_180` is a single 2D sinogram, shape (n_angles, width),
        covering a 0-180 degree angular range.

        `start`/`stop` bound the search range in column-index units.

        Returns the estimated center of rotation, in the same units.
        """


_REGISTRY: dict[str, CalibrationBackend] = {}


def register_backend(cls):
    """
    Decorator: instantiates a CalibrationBackend subclass and adds it
    to the registry when its module is imported.
    """
    instance = cls()
    _REGISTRY[instance.name] = instance
    return cls


def get_backend(name: str) -> CalibrationBackend:
    """Look up a registered backend by name. Raises KeyError, listing
    registered names, if `name` is not found."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown calibration backend '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def clear_registry():
    """Empty the registry. Used by tests."""
    _REGISTRY.clear()
