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
    # True = the projections are already flat/dark corrected (e.g. ILL
    # neutron data): skip the flat/dark division. See
    # preprocessing.build_reference(already_normalized=...).
    projections_normalized: bool = False
    dark_offset: Optional[float] = None
    zinger_removal: bool = False
    zinger_threshold: Optional[float] = None
    zinger_size: Optional[int] = None
    ring_correction: bool = False
    ring_method: str = "sorting"
    ring_options: dict = field(default_factory=dict)         # algorithm-specific, code-only
    # Phase retrieval (propagation-based phase contrast), after zinger removal
    # and before ring removal -- see preprocessing/phase.py. When on, delta/beta
    # has no default (it depends on the material); the other numbers fall back to
    # the project's metadata: energy -> 'energy', distance -> 'propagation_distance',
    # pixel size -> a real 'voxel_size' (not the silent 1.0 um default).
    phase_retrieval: bool = False
    phase_method: str = "paganin"
    phase_delta_beta: Optional[float] = None     # delta/beta of the sample material
    phase_distance_mm: Optional[float] = None     # sample-to-detector; None = metadata
    phase_energy_kev: Optional[float] = None      # None = metadata
    phase_pixel_um: Optional[float] = None        # effective pixel size at the sample; None = metadata
    phase_options: dict = field(default_factory=dict)         # method-specific, code-only
    # Beam hardening correction (polynomial linearization of the attenuation),
    # after ring removal -- see preprocessing/beam_hardening.py. The coefficients
    # (c1, c2, c3, ...) of  p_corrected = c1*p + c2*p^2 + c3*p^3 + ... have no
    # default: (1, 0, 0) changes nothing. bh_options: {"max_attenuation": ...}.
    beam_hardening: bool = False
    bh_coefficients: Optional[tuple] = None
    bh_options: dict = field(default_factory=dict)            # code-only

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
    # None = the default for the project's geometry (reconstruction.default_backend(): gridrec for parallel
    # beam, astra_fdk on the GPU for cone beam), settled by PipelineRun.complete() once the geometry is known.
    reconstruction_backend: Optional[str] = None
    device: Optional[str] = None
    reconstruction_options: dict = field(default_factory=dict)  # backend-specific, code-only
    # Reconstruction filter (ramp, hann, ...), see ReconstructionBackend.filters. None = the backend's
    # default. Ignored by backends without a filter step (iterative ones).
    reconstruction_filter: Optional[str] = None
    # Rotate the reconstructed volume by this many degrees about the scan
    # axis (every xy slice, uniformly). None/0.0 = no rotation. See
    # reconstruction.rotation.reconstruct_rotated() -- rotates the
    # sinogram's angle axis and reconstructs that when the scan's
    # angular coverage supports it exactly, falling back to rotating
    # the reconstructed volume itself otherwise.
    rotate_deg: Optional[float] = None

    # -- postprocess ------------------------------------------------------------
    crop: Optional[tuple] = None                 # ((z0,z1),(y0,y1),(x0,x1))
    dtype: Literal["float32", "uint16"] = "float32"
    dtype_clip_min: Optional[float] = None
    dtype_clip_max: Optional[float] = None

    # -- export -------------------------------------------------------------------
    export_format: Literal["tiff", "hdf5"] = "hdf5"
    export_path: Optional[str] = None            # template; see resolved_project_id()
    export_dataset_path: str = "/entry/volume/data"
    export_extra_settings: dict = field(default_factory=dict)

    # Reconstruct and write one chunk of rows (slices) at a time
    # instead of holding the whole volume in RAM at once -- see
    # PipelineRun._run_streaming(). Requires center/tilt already
    # resolved (not None), and, for a uint16 export, dtype_clip_min/max
    # already resolved too -- there's no whole-volume min/max to
    # default to a chunk at a time. None = size chunks automatically
    # (orchestration.ram_check.estimate_chunk_rows()); a positive int
    # overrides that.
    stream: bool = False
    stream_chunk_rows: Optional[int] = None

    def backend_options(self) -> dict:
        """The options handed to the reconstruction backend: `reconstruction_options`
        plus the chosen filter (as `filter_name`) when the backend has a filter
        step and `reconstruction_options` does not already name one."""
        from reconstruction import get_backend

        options = dict(self.reconstruction_options)
        if self.reconstruction_backend is None:         # not settled yet: nothing to say about its filter
            return options
        if self.reconstruction_filter and "filter_name" not in options:
            if get_backend(self.reconstruction_backend).filters:
                options["filter_name"] = self.reconstruction_filter
        return options

    def resolve_backend(self, geometry_type: Optional[str]) -> str:
        """Settle an unset reconstruction_backend to the default for `geometry_type`; returns the backend's name."""
        from reconstruction import default_backend

        if self.reconstruction_backend is None:
            self.reconstruction_backend = default_backend(geometry_type)
            if self.reconstruction_backend is None:
                raise ValueError(
                    f"no usable {geometry_type or 'parallel'}-beam reconstruction backend is installed "
                    "(see doc/RECONSTRUCTION.md for what each needs)"
                )
        return self.reconstruction_backend

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
