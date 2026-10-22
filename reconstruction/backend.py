"""
ReconstructionBackend: the plugin contract for reconstruction backends,
and the registry backends register with. Backend selection is always
explicit: a name looked up via get_backend(), never auto-detected.

Geometry describes the acquisition geometry a sinogram was taken with.
"""
from __future__ import annotations

import importlib.util
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
        is_iterative:         whether this backend actually uses
                               **options["num_iter"] (SIRT/SART/ART/CGLS)
                               -- direct-inversion backends (FBP, gridrec,
                               BPF, DFI, plain BP) ignore it, and the GUI
                               only shows an iteration-count control when
                               this is True.
        requires:             names of the Python modules this backend
                               needs at run time, e.g. ("algotom",). The
                               backend stays registered without them, but
                               is_available() is False, so the GUI leaves
                               it out of the algorithm list instead of
                               offering a choice that can only fail.
    """

    name: str
    supported_geometries: set
    is_iterative: bool = False
    requires: tuple = ()

    def missing_requirements(self) -> list:
        """The modules from `requires` that cannot be imported right now."""
        return [m for m in self.requires if importlib.util.find_spec(m) is None]

    def is_available(self) -> bool:
        return not self.missing_requirements()

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

    def actual_device(self, device: Optional[str] = None) -> str:
        """
        "gpu" or "cpu": the device a reconstruct() call with this same
        `device` hint actually ends up using, accounting for real
        hardware availability and this backend's own fallback logic --
        not just echoing the hint back. `device=None` (the "try GPU,
        fall back to CPU" default most backends use) is exactly the
        case where the hint alone doesn't tell you what happened;
        call this to report it accurately, e.g. in a status message.

        Base implementation: always "cpu" -- correct for any backend
        with no GPU path of its own. A backend that does have one (see
        AlgotomFbpBackend) overrides this with a real hardware check.
        """
        return "cpu"


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


# Order of the algorithm list in the GUI: the general-purpose analytic methods first,
# then the rest of the built-ins; backends not named here (yours) follow alphabetically.
_DISPLAY_ORDER = (
    "algotom_fbp", "gridrec", "algotom_bpf", "algotom_dfi", "algotom_bp",
    "algotom_sirt", "algotom_sart", "algotom_art", "algotom_cgls", "algotom_gridrec",
)


def list_backends(available_only: bool = True) -> list:
    """Registered backend names, in display order. With available_only (the
    default), backends whose required packages are not installed are left out."""
    names = [n for n, b in _REGISTRY.items() if not available_only or b.is_available()]
    known = [n for n in _DISPLAY_ORDER if n in names]
    return known + sorted(n for n in names if n not in _DISPLAY_ORDER)


def clear_registry():
    """Empty the registry. Used by tests."""
    _REGISTRY.clear()
