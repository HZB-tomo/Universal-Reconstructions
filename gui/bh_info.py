"""
What the Refinements tab shows about the beam hardening correction: how the
current coefficients change a few attenuation values, and whether they are
usable. Qt-free.
"""
from __future__ import annotations

from dataclasses import dataclass

from orchestration import ProjectConfig
from preprocessing.beam_hardening import (
    DEFAULT_MAX_ATTENUATION, MIN_VALID_ATTENUATION, corrected_attenuation, valid_attenuation_limit,
)

DEFAULT_COEFFICIENTS = (1.0, 0.0, 0.0)
SAMPLE_ATTENUATIONS = (1.0, 2.0, 3.0)


@dataclass(frozen=True)
class BeamHardeningSummary:
    coefficients: tuple      # always at least three values (missing ones are 0)
    ok: bool                 # on and usable
    text: str


def padded(coefficients) -> tuple:
    values = list(coefficients) if coefficients else list(DEFAULT_COEFFICIENTS)
    while len(values) < 3:
        values.append(0.0)
    return tuple(float(v) for v in values)


def summarize_beam_hardening(config: ProjectConfig) -> BeamHardeningSummary:
    coefficients = padded(config.bh_coefficients)
    if not config.beam_hardening:
        return BeamHardeningSummary(
            coefficients, False,
            "Off. For a beam with many energies (laboratory tube, white beam): raises the middle of "
            "uniform objects that look too dark (cupping).",
        )
    max_attenuation = float(config.bh_options.get("max_attenuation", DEFAULT_MAX_ATTENUATION))
    limit = valid_attenuation_limit(coefficients, max_attenuation)
    if limit < min(MIN_VALID_ATTENUATION, max_attenuation):
        return BeamHardeningSummary(
            coefficients, False,
            f"Not usable: these coefficients stop increasing at attenuation {limit:.2f}, so different "
            "thicknesses would look the same. Reduce c2/c3.",
        )
    mapped = corrected_attenuation(SAMPLE_ATTENUATIONS, coefficients, max_attenuation)
    examples = ", ".join(f"{p:g} -> {m:.3f}" for p, m in zip(SAMPLE_ATTENUATIONS, mapped))
    return BeamHardeningSummary(
        coefficients, True,
        f"Measured attenuation -> corrected: {examples}. Raise c2 until the middle of a uniform object "
        "is as bright as its edge in a preview.",
    )
