"""
preprocess_rows(): the one place that turns a block of detector rows of a
project into reconstruction-ready sinograms:

    flat/dark normalization -> zinger removal -> phase retrieval -> ring removal -> beam hardening

Used by PipelineRun.preprocess(), by the streaming loop (a chunk at a time) and
by the GUI's single-slice previews, so they cannot drift apart.

Phase retrieval is the reason this is more than a few calls: it filters across
neighbouring rows, so the block is read with `halo` extra rows on each side
(clipped at the detector edge, where the filter pads by repeating the edge)
and the halo is cut off again afterwards. The answer for a block is then the
same as the answer for the same rows of the whole image.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.data_interface import DataInterface
from core.project import ProjectObject
from orchestration.config import PipelineStageError, ProjectConfig
from preprocessing import (
    FlatDarkReference, normalize_rows, process_volume, remove_ring_artifacts, remove_zinger,
)
from preprocessing.beam_hardening import (
    DEFAULT_MAX_ATTENUATION, MIN_VALID_ATTENUATION, correct_beam_hardening, valid_attenuation_limit,
)
from preprocessing.phase import PhaseMethod, PhaseParams, get_phase_method


@dataclass(frozen=True)
class PhaseSetup:
    method: PhaseMethod
    params: PhaseParams
    options: dict
    halo: int      # rows needed above and below a block


def _number(value) -> Optional[float]:
    """Metadata values arrive as numbers from a file but as text from a hand-written INI."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# What phase retrieval needs, in the order the GUI shows it: name -> where to set it.
PHASE_INPUT_HINTS = {
    "delta_over_beta": "ProjectConfig.phase_delta_beta (material dependent; no safe default)",
    "distance_mm": "phase_distance_mm, or the 'propagation_distance' metadata field (sample-to-detector, mm)",
    "energy_kev": "phase_energy_kev, or the 'energy' metadata field (keV)",
    "pixel_um": "phase_pixel_um, or a real pixel size in Basic settings / the file ('voxel_size', um)",
}


def phase_inputs(project: Optional[ProjectObject], config: ProjectConfig) -> dict:
    """
    Where each physical input of phase retrieval comes from right now:
    {name: (value or None, origin)}, origin one of "config" (set explicitly), "file"
    (read from the project's file), "user" (typed in as a completion answer) or "" (not
    known). Config wins over metadata. A pixel size that is only the silent default of
    1.0 um is reported as unknown. Works without a project (metadata then counts as unknown).
    """
    di = DataInterface(project) if project is not None else None

    def from_metadata(key):
        if di is None:
            return None, ""
        source = di.get_metadata_source(key)
        if source == "default":
            return None, ""
        value = _number(di.get_metadata(key))
        return (value, source or "") if value is not None else (None, "")

    def pick(configured, key):
        if configured is not None:
            return configured, "config"
        return from_metadata(key)

    return {
        "delta_over_beta": (config.phase_delta_beta, "config") if config.phase_delta_beta is not None else (None, ""),
        "distance_mm": pick(config.phase_distance_mm, "propagation_distance"),
        "energy_kev": pick(config.phase_energy_kev, "energy"),
        "pixel_um": pick(config.phase_pixel_um, "voxel_size"),
    }


def resolve_phase(project: ProjectObject, config: ProjectConfig) -> Optional[PhaseSetup]:
    """
    None when phase retrieval is off. Otherwise the method and its physical
    parameters: each comes from the config if set there, else from the project's
    metadata (energy, propagation_distance, voxel_size). Raises PipelineStageError,
    naming everything that is missing, instead of guessing physics.
    """
    if not config.phase_retrieval:
        return None

    values = phase_inputs(project, config)
    missing = [f"{name}: set {PHASE_INPUT_HINTS[name]}" for name, (value, _origin) in values.items() if value is None]
    if missing:
        raise PipelineStageError("preprocess", "phase retrieval is on but needs more information -- " + "; ".join(missing))

    try:
        method = get_phase_method(config.phase_method)
        params = PhaseParams(**{name: value for name, (value, _origin) in values.items()})
    except (KeyError, ValueError) as e:
        raise PipelineStageError("preprocess", f"phase retrieval: {e}") from e
    options = dict(config.phase_options)
    return PhaseSetup(method, params, options, method.halo_rows(params, **options))


@dataclass(frozen=True)
class BeamHardeningSetup:
    coefficients: tuple
    max_attenuation: float


def resolve_beam_hardening(config: ProjectConfig) -> Optional[BeamHardeningSetup]:
    """None when the correction is off; otherwise its validated coefficients. Raises
    PipelineStageError if they are missing or the polynomial is not usable."""
    if not config.beam_hardening:
        return None
    if not config.bh_coefficients:
        raise PipelineStageError(
            "preprocess", "beam hardening correction is on but has no coefficients -- set bh_coefficients "
            "(c1, c2, c3, ... of p_corrected = c1 p + c2 p^2 + c3 p^3); (1, 0, 0) changes nothing",
        )
    max_attenuation = float(config.bh_options.get("max_attenuation", DEFAULT_MAX_ATTENUATION))
    coefficients = tuple(float(c) for c in config.bh_coefficients)
    limit = valid_attenuation_limit(coefficients, max_attenuation)
    if limit < min(MIN_VALID_ATTENUATION, max_attenuation):
        raise PipelineStageError(
            "preprocess", f"beam hardening coefficients {list(coefficients)} stop increasing at attenuation "
            f"{limit:.2f} (need at least {MIN_VALID_ATTENUATION:g}); reduce the higher-order terms",
        )
    return BeamHardeningSetup(coefficients, max_attenuation)


def preprocess_rows(
    project: ProjectObject, config: ProjectConfig, reference: FlatDarkReference, rows,
    phase: Optional[PhaseSetup] = None,
) -> np.ndarray:
    """
    Normalized, corrected sinograms for `rows` (a slice, or anything numpy accepts as a row
    index when phase retrieval is off): shape (n_angles, n_rows, width), float32.
    `phase` is resolve_phase()'s result; pass it to avoid resolving it for every chunk.
    """
    if phase is None:
        phase = resolve_phase(project, config)

    if phase is None:
        block = normalize_rows(project, rows, reference)
        keep = None
    else:
        if not isinstance(rows, slice):
            raise ValueError("phase retrieval needs `rows` to be a slice")
        n_rows = DataInterface(project).shape_of("projections")[1]
        start, stop, _ = rows.indices(n_rows)
        lo, hi = max(0, start - phase.halo), min(n_rows, stop + phase.halo)
        block = normalize_rows(project, slice(lo, hi), reference)
        keep = (start - lo, stop - lo)

    if config.zinger_removal:
        block = process_volume(
            block, remove_zinger,
            config.zinger_threshold if config.zinger_threshold is not None else 0.08,
            size=config.zinger_size if config.zinger_size is not None else 2,
        )
    if phase is not None:
        block = phase.method.retrieve(block, phase.params, **phase.options)
        block = np.ascontiguousarray(block[:, keep[0]:keep[1], :]) if keep is not None else block
    if config.ring_correction:
        block = process_volume(block, remove_ring_artifacts, method=config.ring_method, **config.ring_options)
    hardening = resolve_beam_hardening(config)
    if hardening is not None:
        block = correct_beam_hardening(block, hardening.coefficients, max_attenuation=hardening.max_attenuation)
    return block
