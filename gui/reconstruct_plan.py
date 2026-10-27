"""
Qt-free decision logic for "Reconstruct all slices": how much RAM an
in-memory run needs, whether a streamed-to-HDF5 run is allowed, and the
config that run uses. Kept apart from the dialog/panel so it can be
tested without a display. See AdjustmentPanel._reconstruct_all().
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np

from orchestration import ProgressEvent, ProjectConfig, ram_check
from reconstruction import StreamingVolumeWriter


@dataclass(frozen=True)
class ReconstructPlan:
    needed_bytes: int       # estimated peak for an in-memory run, safety margin included
    available_bytes: int
    fits_in_ram: bool
    can_stream: bool        # streaming writes independent parallel-beam slices; see can_stream_geometry()
    stream_blocker: str     # why can_stream is False; "" otherwise


def can_stream_geometry(geometry_type: Optional[str]) -> bool:
    """
    Streaming reconstructs each chunk of rows from just that chunk's
    sinograms, which holds only for parallel-beam: every detector row
    is an independent 2D problem. A cone-beam slice also needs the
    neighbouring detector rows its rays cross, so row-chunking would
    need overlapping row windows per slab -- not implemented (and no
    cone-beam backend is registered yet).
    """
    return geometry_type == "parallel"


def make_plan(n_angles: int, n_rows: int, width: int, geometry_type: Optional[str]) -> ReconstructPlan:
    raw = ram_check.estimate_inmemory_run_bytes(n_angles, n_rows, width)
    available = ram_check.available_bytes()
    can_stream = can_stream_geometry(geometry_type)
    if can_stream:
        blocker = ""
    elif geometry_type:
        blocker = f"streaming to disk needs parallel-beam geometry (this project is '{geometry_type}')"
    else:
        blocker = "streaming to disk needs parallel-beam geometry, and this project's geometry type isn't set yet (Basic settings)"
    return ReconstructPlan(
        needed_bytes=int(raw * ram_check.SAFETY_FACTOR),
        available_bytes=available,
        fits_in_ram=ram_check.fits_in_available_ram(raw),
        can_stream=can_stream,
        stream_blocker=blocker,
    )


def rows_for_export(export_rows: Optional[tuple[int, int]], n_rows: int) -> tuple[Optional[slice], int]:
    """
    Save settings' From/To slice (both inclusive, ProjectSession.export_rows)
    as (config.rows, number of slices). None means every row.
    """
    if export_rows is None or export_rows == (0, n_rows - 1):
        return None, n_rows
    start, end = max(0, export_rows[0]), min(n_rows - 1, export_rows[1])
    return slice(start, end + 1), max(end - start + 1, 0)


def describe_run(config: ProjectConfig, n_rows_total: int, n_selected: int) -> str:
    """The 'what will be reconstructed' lines of the destination dialog."""
    rows = config.rows
    if rows is None:
        slices = f"Slices: all {n_rows_total}"
    else:
        first, last, _ = rows.indices(n_rows_total)
        slices = f"Slices: {first} to {last - 1} ({n_selected} of {n_rows_total})"
    if config.dtype == "uint16" and (config.dtype_clip_min is None or config.dtype_clip_max is None):
        output = (
            "Output: uint16 in memory (scaled to the volume's own range); an HDF5 file is written as float32 "
            "unless Clip min/max are set in Save settings"
        )
    else:
        output = f"Output: {config.dtype}"
    return f"{slices}\n{output}"


def streaming_config(config: ProjectConfig, path: str) -> tuple[ProjectConfig, str]:
    """
    A copy of `config` set up to stream into the HDF5 file at `path`:
    config.rows (the Save settings slice range, set by the caller) and
    config.dtype are kept; crop is not applied. Returns (config, note);
    `note` is non-empty when a setting had to change. A uint16 export without explicit clip bounds
    can't stream (there's no whole-volume min/max to discover a chunk
    at a time), so it is written as float32 instead.
    """
    note = ""
    dtype = config.dtype
    if dtype == "uint16" and (config.dtype_clip_min is None or config.dtype_clip_max is None):
        dtype = "float32"
        note = "written as float32 (uint16 needs Clip min/max set in Save settings to stream)"
    return replace(
        config, crop=None, stream=True, export_format="hdf5", export_path=str(path), dtype=dtype,
    ), note


def describe_error(exc: BaseException) -> str:
    """A MemoryError's own message is often empty; say what happened."""
    if isinstance(exc, MemoryError):
        return (
            "out of memory" + (f" ({exc})" if str(exc) else "")
            + " -- close other programs, or choose 'Write to HDF5 file' so the volume is streamed to disk"
        )
    return str(exc) or type(exc).__name__


def unique_path(path) -> str:
    """`path`, or `name_1.h5`, `name_2.h5`, ... if it already exists --
    saving never overwrites an existing file."""
    path = Path(path)
    candidate, n = path, 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        n += 1
    return str(candidate)


def save_ram_volume(
    volume: np.ndarray, path: str, settings: Optional[dict] = None, dataset_path: str = "/entry/volume/data",
    progress=None, project_id: str = "", chunk_slices: int = 64,
) -> str:
    """Writes an in-memory (n_slices, height, width) volume to a new
    HDF5 file, a chunk of slices at a time so `progress` (ProgressEvent
    callback) can report slices saved. Returns the path actually used."""
    path = unique_path(path)
    n_slices, height, width = volume.shape
    with StreamingVolumeWriter(
        path, format="hdf5", total_slices=n_slices, height=height, width=width,
        dtype=volume.dtype, dataset_path=dataset_path, settings=settings,
    ) as writer:
        if progress is not None:
            progress(ProgressEvent(project_id, "Saving", 0, n_slices))
        for start in range(0, n_slices, chunk_slices):
            stop = min(start + chunk_slices, n_slices)
            writer.write_chunk(volume[start:stop], z_offset=start)
            if progress is not None:
                progress(ProgressEvent(project_id, "Saving", stop, n_slices))
    return path


class RememberedPaths:
    """Output locations chosen earlier this session (in memory only),
    so later reconstructions -- including a future 'all selected
    projects' one -- can start from the same folder."""

    def __init__(self) -> None:
        self.volume_dir: Optional[str] = None

    def remember_volume_path(self, path: str) -> None:
        self.volume_dir = str(Path(path).parent)

    def default_volume_path(self, config: ProjectConfig) -> str:
        name = f"{config.resolved_project_id()}_volume.h5"
        if self.volume_dir is not None:
            return str(Path(self.volume_dir) / name)
        return str(Path(config.path).parent / name)
