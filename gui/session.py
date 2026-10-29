"""
ProjectSession: one imported project as the GUI tracks it. Pairs a
ProjectConfig with the PipelineRun driving it through the backend, plus
GUI-only state that belongs in neither of those.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from orchestration import PipelineRun, ProjectConfig


@dataclass
class ProjectSession:
    config: ProjectConfig
    run: Optional[PipelineRun] = None
    checked: bool = False
    # Set by MainWindow._reconstruct_selected() on success: run_batch()
    # builds its own internal PipelineRun per config and doesn't hand
    # back the volume, so session.run.volume (what _status_of() checks
    # for a single-project reconstruct) never gets populated that way --
    # this is the equivalent signal for the batch path.
    exported: bool = False
    # None until AdjustmentPanel knows this project's row count and
    # resolves it to the middle row -- see _refresh_from_session().
    preview_row: Optional[int] = None
    error: Optional[str] = None
    # Save settings' "from slice"/"to slice" range, BOTH INCLUSIVE
    # (matching how a person says "slices 100 to 500"), and its crop
    # rectangle -- (y0, y1, x0, x1), HALF-OPEN, matching Python slicing
    # directly (this one feeds straight into config.crop with no
    # conversion). Neither has any meaning to orchestration on its own
    # -- MainWindow._reconstruct_selected() translates both into
    # config.rows/config.crop right before exporting; AdjustmentPanel's
    # own preview buttons never export and so never consult either.
    # None means "everything"/"no crop", same as the config fields they
    # translate into.
    export_rows: Optional[tuple[int, int]] = None
    export_crop_yx: Optional[tuple[int, int, int, int]] = None
    # The project's full reconstruction, for the project panel's
    # "Reconstructed" row: an HDF5 file it was streamed/exported to, or
    # (Reconstruct all slices, "Keep in memory only") the array itself.
    # Single-slice previews set neither.
    volume_path: Optional[str] = None
    volume_dataset: str = "/entry/volume/data"
    ram_volume: Optional[np.ndarray] = None
    # Where the viewer was left for this project, per view ("projections" /
    # "reconstruction"): slicing direction, slice number, zoom. See
    # ImageViewer.view_state() and MainWindow._remember_view().
    view_states: dict = field(default_factory=dict)
    # Center-of-rotation scan (Automatic adjustments tab): the metric, the range of trial centers as shifts
    # from the detector middle, how many points, and the last scan (a calibration.CenterScan). None = defaults.
    cor_metric: Optional[str] = None
    cor_range: Optional[tuple] = None
    cor_points: Optional[int] = None
    cor_range_user: bool = False      # the range was typed by the person: the quick estimate then searches it
    cor_scan: Optional[object] = None

    @property
    def project_id(self) -> str:
        return self.config.resolved_project_id()

    @property
    def imported(self) -> bool:
        return self.run is not None and self.run.project is not None


def unique_project_id(candidate: str, existing: set) -> str:
    """
    ProjectConfig.resolved_project_id() is just the path stem, so two
    files named the same in different folders collide. Disambiguates
    `candidate` against `existing` (a set of ids already in use in the
    project panel).
    """
    if candidate not in existing:
        return candidate
    n = 2
    while f"{candidate}_{n}" in existing:
        n += 1
    return f"{candidate}_{n}"
