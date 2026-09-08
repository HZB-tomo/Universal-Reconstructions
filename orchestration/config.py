"""
ProjectConfig: everything one project's run through the pipeline needs
that isn't derivable from the data file itself. One dataclass, grouped
by pipeline stage; every field has a default meaning "use the
automatic/library default" so a config can supply as little or as much
as it needs to.

PipelineStageError is the one exception type every orchestration.pipeline
stage raises on failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional


@dataclass
class ProjectConfig:
    # -- identity ---------------------------------------------------------
    project_id: Optional[str] = None            # default: stem of `path`

    # -- import -------------------------------------------------------------
    path: str = ""
    format: Literal["hdf5", "tiff"] = "hdf5"
    storage_mode: Literal["ram", "disk"] = "ram"
    plugin_name: Optional[str] = None            # disambiguate when >1 plugin matches

    # -- completion -----------------------------------------------------------
    completion_answers: dict = field(default_factory=dict)   # e.g. {"geometry_type": "parallel"}

    # -- preprocessing ----------------------------------------------------------
    rows: Optional[Any] = None                   # None = every row
    reference_reducer: Literal["mean", "median"] = "mean"
    dark_offset: Optional[float] = None
    zinger_removal: bool = False
    zinger_threshold: Optional[float] = None
    zinger_size: Optional[int] = None
    ring_correction: bool = False
    ring_method: str = "sorting"
    ring_options: dict = field(default_factory=dict)         # algorithm-specific, code-only

    # -- calibration ------------------------------------------------------------
    calibration_backend: str = "algotom"
    center: Optional[float] = None               # None = auto find_center()
    tilt_deg: Optional[float] = None              # None = auto find_tilt()
    tilt_num_rows: Optional[int] = None
    tilt_margin: Optional[int] = None
    tilt_outlier_tolerance: Optional[float] = None

    # -- reconstruction -----------------------------------------------------------
    # geometry_type, source_to_detector_distance, source_to_object_distance,
    # and detector_pixel_size are NOT config fields: they are already
    # core/known_fields.py entries, resolved onto project.metadata by
    # the completion stage (via completion_answers above, or the
    # file/derive/default strategies already built for them) before
    # reconstruct() runs. Duplicating them here would let a project's
    # answer disagree with itself.
    reconstruction_backend: str = "algotom_fbp"
    device: Optional[str] = None
    reconstruction_options: dict = field(default_factory=dict)  # backend-specific, code-only

    # -- postprocess ------------------------------------------------------------
    crop: Optional[tuple] = None                 # ((z0,z1),(y0,y1),(x0,x1))
    dtype: Literal["float32", "uint16"] = "float32"
    dtype_clip_min: Optional[float] = None
    dtype_clip_max: Optional[float] = None

    # -- export -------------------------------------------------------------------
    export_format: Literal["tiff", "hdf5"] = "hdf5"
    export_path: Optional[str] = None            # template; see resolved_project_id()
    export_dataset_path: str = "/volume"
    export_extra_settings: dict = field(default_factory=dict)

    def resolved_project_id(self) -> str:
        """project_id if set, else the stem of `path` (the data file's
        or folder's own name, which usually already identifies the
        project)."""
        if self.project_id:
            return self.project_id
        return Path(self.path).stem


class PipelineStageError(Exception):
    """
    Raised by any orchestration.pipeline.PipelineRun stage on failure.
    `stage` names which stage method raised it; `fields`, if given, is
    the specific list of metadata fields that could not be resolved
    (only meaningful for a "complete" stage failure).
    """

    def __init__(self, stage: str, message: str, fields: Optional[list] = None):
        self.stage = stage
        self.fields = fields
        super().__init__(f"[{stage}] {message}")
