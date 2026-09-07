"""
ReconstructionBackend: the plugin contract for reconstruction backends,
and the registry backends register with. Backend selection is always
explicit: a name looked up via get_backend(), never auto-detected.

Geometry describes the acquisition geometry a sinogram was taken with.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass
class Geometry:
    """
    Acquisition geometry for a sinogram, and the extra parameters
    cone-beam needs.
    """

    geometry_type: Literal["parallel", "cone"]
    source_to_detector_distance: Optional[float] = None  # mm, cone only
    source_to_object_distance: Optional[float] = None  # mm, cone only
    detector_pixel_size: Optional[float] = None  # um; when set, reconstruct()
    # divides the backend's output by this value to produce attenuation
    # coefficients in 1/um instead of "per pixel" units.


class ReconstructionBackend(ABC):
    """
    Base class for a reconstruction backend. Subclasses implement
    exactly one method: reconstruct().

    Class attributes to set on a subclass:
        name:                 a short unique string, e.g. "algotom_fbp".
                               Used for lookup via get_backend().
        supported_geometries: the Geometry.geometry_type values this
                               backend supports, e.g. {"parallel"} or
                               {"parallel", "cone"}.
    """

    name: str
    supported_geometries: set

    @abstractmethod
    def reconstruct(
        self,
        sinograms: np.ndarray,
        angles: np.ndarray,
        center,
        geometry: Geometry,
        device: Optional[str] = None,
        **options,
    ) -> np.ndarray:
        """
        `sinograms` is a 3D stack, shape (n_angles, n_slices, n_columns),
        angle-first. A single requested slice arrives as
        (n_angles, 1, n_columns); reconstruction/api.py handles wrapping
        and unwrapping at the public API boundary.

        Returns a stack shaped (n_slices, height, width), slice-first.

        Return values are attenuation coefficients in "per pixel"
        units. Do not scale by `geometry.detector_pixel_size`;
        api.reconstruct() does that centrally. Any numeric dtype is
        acceptable; api.reconstruct() casts the result to float32.

        `center` is a single scalar, shared by every sinogram in the
        stack, or an array-like of one value per sinogram.

        `device` is a hint ("cpu"/"gpu"), not a guarantee. A backend may
        honor it, ignore it, or use its own default when `device` is
        None.

        `**options` are algorithm-specific and defined by each backend.
        """


_REGISTRY: dict[str, ReconstructionBackend] = {}


def register_backend(cls):
    """
    Decorator: instantiates a ReconstructionBackend subclass and adds
    it to the registry when its module is imported.
    """
    instance = cls()
    _REGISTRY[instance.name] = instance
    return cls


def get_backend(name: str) -> ReconstructionBackend:
    """Look up a registered backend by name. Raises KeyError, listing
    registered names, if `name` is not found."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown reconstruction backend '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def clear_registry():
    """Empty the registry. Used by tests."""
    _REGISTRY.clear()
