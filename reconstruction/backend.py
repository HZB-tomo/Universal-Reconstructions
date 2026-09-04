"""
The reconstruction plugin contract (ReconstructionBackend) and the
registry plugins sign up with -- the same idiom as data_import/plugin.py,
deliberately without a Signature/matches() equivalent. A vendor plugin
has to be auto-detected because the software doesn't know in advance
which vendor wrote a given file; a reconstruction backend doesn't have
that problem -- you already know you want algotom, or ASTRA, or
whatever gets added later, so selection is always explicit
(get_backend("algotom_fbp")), never inferred.

Geometry is the one other thing every backend needs: which kind
(parallel/cone/...) and, for cone, the extra distances/pixel size --
all fields that already exist in core/known_fields.py, so this is just
a small bundle of them for passing around outside a ProjectObject, not
a new source of truth.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass
class Geometry:
    """
    Which acquisition geometry a sinogram was taken with, and the
    extra parameters cone-beam needs. Field names match
    core/known_fields.py's registry exactly -- a caller working from a
    ProjectObject builds one of these from project.metadata, this class
    itself has no dependency on ProjectObject at all.
    """

    geometry_type: Literal["parallel", "cone"]
    source_to_detector_distance: Optional[float] = None  # mm, cone only
    source_to_object_distance: Optional[float] = None  # mm, cone only
    detector_pixel_size: Optional[float] = None  # um -- when set, reconstruct()
    # divides the raw backend output by this value to convert attenuation
    # coefficients from "per pixel" to real 1/um units; see api.reconstruct()
    # and doc/RECONSTRUCTION.md's "Units" section.


class ReconstructionBackend(ABC):
    """
    Base class every reconstruction plugin subclasses. Exactly one
    method to implement: reconstruct(). Whether that loops internally,
    calls into a library's own native batch API, uses CPU or GPU, is
    entirely this method's own business -- see reconstruction/api.py's
    module docstring for why the contract is deliberately silent about
    HOW a backend gets from sinograms to images, only WHAT goes in and
    out.

    Class attributes to set on your subclass:
        name:                 a short unique string, e.g. "algotom_fbp".
                               Used for lookup via get_backend().
        supported_geometries: which Geometry.geometry_type values this
                               backend can actually handle, e.g.
                               {"parallel"} or {"parallel", "cone"}.
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
        `sinograms` is always a 3D stack, (n_angles, n_slices, n_columns)
        -- angle-first, matching preprocessing.normalize_rows()'s own
        output convention exactly, so a caller can hand its result
        straight to reconstruct() with no axis reshuffling in between.
        Even a single requested slice arrives this way, wrapped to
        (n_angles, 1, n_columns) -- reconstruction/api.py handles that
        wrapping/unwrapping at the public API boundary, so every backend
        only ever has to deal with one shape.

        Returns a stack shaped (n_slices, height, width) -- slice-first,
        matching this codebase's own volume convention instead (e.g.
        tests/fixtures/realistic_phantom/phantom.py's `(height, width, width)`
        phantom volume) -- input and output are different kinds of
        stacks (sinograms vs. reconstructed images), so matching two
        different existing conventions here is deliberate, not
        inconsistent.

        Return values are attenuation coefficients in "per pixel" units
        -- i.e. whatever your underlying algorithm naturally produces
        assuming unit pixel spacing (this is what every FBP/iterative
        library actually computes internally). Do NOT scale by
        `geometry.detector_pixel_size` yourself -- api.reconstruct()
        does that centrally, once, for every backend, so this method
        never needs to know or care whether a pixel size was given. Any
        numeric dtype is fine to return; api.reconstruct() also casts
        the final result to float32, so this method doesn't need to
        worry about that either.

        `center` is either a single scalar (the same center for every
        sinogram in the stack) or an array-like of one value per
        sinogram -- e.g. a calibration search that pre-shifts N copies
        of the same row by different candidate amounts and reconstructs
        them all with one shared center, versus a caller supplying a
        genuinely different, already-known center per physical row.

        `device` is a hint ("cpu"/"gpu"), not a guarantee -- a backend
        is free to honor it, ignore it, or use its own default if
        `device` is None. This contract makes no promise about
        automatic cost-based CPU/GPU selection; a backend that wants
        that has to implement it itself.

        `**options` are algorithm-specific (filter name, iteration
        count, whatever this backend's underlying library needs) --
        entirely this backend's own to define and document.
        """


_REGISTRY: dict[str, ReconstructionBackend] = {}


def register_backend(cls):
    """
    Decorator: put @register_backend above a ReconstructionBackend
    subclass and it's automatically instantiated and added to the
    registry as soon as its module is imported -- same pattern as
    data_import/plugin.py's @register_plugin.
    """
    instance = cls()
    _REGISTRY[instance.name] = instance
    return cls


def get_backend(name: str) -> ReconstructionBackend:
    """Look up a registered backend by name. Raises KeyError, listing
    what IS registered, if `name` isn't found -- selection here is
    always explicit, never auto-detected."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown reconstruction backend '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def clear_registry():
    """Empty the registry. Exists for tests, mirroring data_import.plugin.clear_registry()."""
    _REGISTRY.clear()
