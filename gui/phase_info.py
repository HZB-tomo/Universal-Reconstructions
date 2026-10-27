"""
What the Refinements tab shows about phase retrieval: the value of each input
and where it came from, and a one-line summary of what the current settings
do (smoothing length, halo) or what is still missing. Qt-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.project import ProjectObject
from orchestration import ProjectConfig
from orchestration.preprocess import phase_inputs
from preprocessing.phase import PhaseParams, get_phase_method

LABELS = {
    "delta_over_beta": "delta/beta",
    "distance_mm": "propagation distance",
    "energy_kev": "energy",
    "pixel_um": "pixel size",
}
ORIGIN_TEXT = {"config": "set here", "file": "from the file", "user": "typed in", "": "not set"}


@dataclass(frozen=True)
class PhaseSummary:
    values: dict           # name -> (value or None, origin), see orchestration.preprocess.phase_inputs()
    missing: list          # names with no value
    ok: bool               # phase retrieval is on and fully specified
    text: str


def summarize_phase(project: Optional[ProjectObject], config: ProjectConfig) -> PhaseSummary:
    values = phase_inputs(project, config)
    missing = [name for name, (value, _origin) in values.items() if value is None]
    if not config.phase_retrieval:
        return PhaseSummary(values, missing, False, "Off. Tick the box to remove phase-contrast fringes before reconstructing.")
    if missing:
        names = ", ".join(LABELS[name] for name in missing)
        return PhaseSummary(values, missing, False, f"Still needed: {names}.")
    try:
        params = PhaseParams(**{name: value for name, (value, _origin) in values.items()})
        method = get_phase_method(config.phase_method)
    except (KeyError, ValueError) as e:
        return PhaseSummary(values, missing, False, f"Cannot run: {e}")
    blur = params.blur_length_um
    halo = method.halo_rows(params, **config.phase_options)
    text = (
        f"Smoothing length {blur:.2f} um ({blur / params.pixel_um:.1f} px); "
        f"{halo} extra rows are read at each edge of a block. "
        "A larger delta/beta or distance removes more noise and blurs more detail."
    )
    return PhaseSummary(values, missing, True, text)
