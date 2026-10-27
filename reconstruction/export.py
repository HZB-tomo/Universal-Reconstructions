"""
save_volume(): writes a reconstructed volume to disk, as a TIFF stack
or as a volume file of the URCT HDF5 standard (reconstruction/volume_file.py,
doc/FILE_STANDARD.md), with an optional settings/provenance dict (backend
used, center, tilt, ring correction, crop, dtype clipping, ...; see
core/project.py's ProjectObject.export_metadata() for pulling acquisition
metadata into the same dict).

StreamingVolumeWriter is save_volume()'s incremental counterpart, for
a caller (PipelineRun._run_streaming()) that reconstructs one chunk of
slices at a time and never holds the whole volume in memory at once --
see its own docstring below.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np

from core import urct_format as fmt
from reconstruction import volume_file


def save_volume(
    volume: np.ndarray,
    path: Union[str, Path],
    format: Literal["tiff", "hdf5"],
    dataset_path: str = fmt.VOLUME,
    settings: Optional[dict] = None,
) -> None:
    """
    Write a reconstructed volume to disk. `volume` is a (n_slices,
    height, width) stack, or a single (height, width) image.

    format="tiff" writes one file per slice into the folder at `path`
    (created if it doesn't exist yet), named `slice_00000.tif`,
    `slice_00001.tif`, ... . This is the same folder-of-files layout
    data_import's TiffBackend reads.

    format="hdf5" writes a volume file of the URCT standard at `path`
    (created, or overwritten if it already exists): the volume at
    `dataset_path` (default /entry/volume/data; chunked in cubes of 64, not
    compressed), the reconstruction record, and the measurement metadata.

    `settings`, if given, is a plain dict recording anything about how
    this volume was produced: reconstruction backend and options,
    center, tilt, whether ring correction ran, crop bounds, dtype-clip
    range, or acquisition metadata from ProjectObject.export_metadata().
    For format="hdf5" the known entries are filed into the standard's
    reconstruction record and measurement groups (see volume_file.py),
    and the whole dict is also kept as JSON in the dataset's
    `reconstruction_settings` attribute; for format="tiff" it is a
    `metadata.json` file alongside the slices. Values must be
    JSON-serializable; numpy scalars and arrays are converted automatically.

    An HDF5 volume records its units (1/cm when settings holds the
    detector_pixel_size, else 1/px) and, for uint16, the scale and offset
    that turn stored values back into physical ones. TIFF stacks record
    nothing of the sort on their own; see reconstruct()'s docstring.
    """
    volume = np.asarray(volume)
    if volume.ndim == 2:
        volume = volume[np.newaxis]

    path = Path(path)
    if format == "tiff":
        _save_tiff_stack(volume, path, settings)
    elif format == "hdf5":
        _save_hdf5(volume, path, dataset_path, settings)
    else:
        raise ValueError(f"Unknown format {format!r}, expected 'tiff' or 'hdf5'")


def _to_jsonable(value):
    """Recursively convert numpy scalars/arrays (anywhere inside dicts
    or lists/tuples) into plain JSON-serializable types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _save_tiff_stack(volume: np.ndarray, folder: Path, settings: Optional[dict]) -> None:
    import tifffile  # imported here so this package stays importable without tifffile installed

    folder.mkdir(parents=True, exist_ok=True)
    for stale in folder.glob("slice_*.tif"):
        stale.unlink()

    n_digits = max(5, len(str(volume.shape[0] - 1)))
    for i, image in enumerate(volume):
        tifffile.imwrite(folder / f"slice_{i:0{n_digits}d}.tif", image)

    if settings:
        (folder / "metadata.json").write_text(json.dumps(_to_jsonable(settings), indent=2))


def _save_hdf5(volume: np.ndarray, path: Path, dataset_path: str, settings: Optional[dict]) -> None:
    f, dataset = volume_file.create_volume_file(path, volume.shape, volume.dtype, settings, dataset_path)
    with f:
        dataset[...] = volume


class StreamingVolumeWriter:
    """
    save_volume()'s incremental counterpart: the output volume's final
    shape (total_slices, height, width) is known and pre-allocated
    upfront, and write_chunk() fills in one range of slices at a time
    -- the caller never needs to hold more than one chunk in memory.

    Chunks may be written in ANY order, not just increasing z_offset:
    for format="hdf5", writing is by absolute index into an
    already-allocated dataset (dataset[z0:z1] = chunk), which is
    exactly as correct done out of order as in order -- HDF5 is
    randomly addressable, not a stream/log. For format="tiff", each
    slice's filename (slice_00042.tif) already carries its own
    absolute index, so files can land on disk in any order too. There
    is no reassembly/reordering step anywhere in this class because
    none is needed.

    `settings` (backend, center, tilt, crop, dtype-clip, ...) is
    written once, at construction, not deferred to close() -- the
    streaming caller already has every value that would go into it
    resolved before the first chunk starts (see
    PipelineRun._run_streaming(): streaming requires center/tilt, and,
    for a uint16 export, dtype_clip_min/max, to already be known --
    there's no whole-volume value to discover partway through).
    """

    def __init__(
        self,
        path: Union[str, Path],
        format: Literal["tiff", "hdf5"],
        total_slices: int,
        height: int,
        width: int,
        dtype,
        dataset_path: str = fmt.VOLUME,
        settings: Optional[dict] = None,
    ):
        self._format = format
        self._blocks: dict = {}
        self._n_digits = max(5, len(str(max(total_slices - 1, 0))))

        if format == "tiff":
            import tifffile  # imported here so this package stays importable without tifffile installed

            self._tifffile = tifffile
            self._folder = Path(path)
            self._folder.mkdir(parents=True, exist_ok=True)
            for stale in self._folder.glob("slice_*.tif"):
                stale.unlink()
            if settings:
                (self._folder / "metadata.json").write_text(json.dumps(_to_jsonable(settings), indent=2))
        elif format == "hdf5":
            self._file, self._dataset = volume_file.create_volume_file(
                path, (total_slices, height, width), dtype, settings, dataset_path,
            )
            # Chunks are 64 slices deep. Writing a few slices into a chunk would read and rewrite it
            # again and again, so the slices are collected into whole 64-slice blocks first (as long
            # as a block is not huge) and each block is written once.
            self._depth = self._dataset.chunks[0]
            block_bytes = self._depth * height * width * np.dtype(dtype).itemsize
            self._buffered = self._depth > 1 and block_bytes <= volume_file.BUFFER_LIMIT
            self._total = total_slices
            self._shape = (height, width)
            self._dtype = np.dtype(dtype)
        else:
            raise ValueError(f"Unknown format {format!r}, expected 'tiff' or 'hdf5'")

    def write_chunk(self, chunk: np.ndarray, z_offset: int) -> None:
        """`chunk` is (n, height, width); written at absolute slice
        indices [z_offset, z_offset + n). See class docstring for why
        z_offset doesn't need to arrive in increasing order."""
        if self._format == "tiff":
            for i, image in enumerate(chunk):
                self._tifffile.imwrite(self._folder / f"slice_{z_offset + i:0{self._n_digits}d}.tif", image)
        elif not self._buffered:
            self._dataset[z_offset:z_offset + chunk.shape[0]] = chunk
        else:
            depth = self._depth
            z, end = z_offset, z_offset + chunk.shape[0]
            while z < end:
                block = z // depth
                start = block * depth
                stop = min(start + depth, self._total)
                part = min(end, stop)
                entry = self._blocks.get(block)
                if entry is None:
                    entry = self._blocks[block] = [np.zeros((stop - start, *self._shape), dtype=self._dtype), 0]
                entry[0][z - start:part - start] = chunk[z - z_offset:part - z_offset]
                entry[1] += part - z
                if entry[1] >= stop - start:
                    self._dataset[start:stop] = entry[0]
                    del self._blocks[block]
                z = part

    def update_record(self, **entries) -> None:
        """Add or replace reconstruction-record entries after the run (timings, speed, ...),
        by their settings names: duration_s, speed_slices_per_s, finished, ..."""
        if self._format == "hdf5":
            volume_file.update_record(self._file, entries)

    def close(self) -> None:
        if self._format == "hdf5":
            for block, (data, _filled) in sorted(self._blocks.items()):   # blocks never completed (a stopped run)
                start = block * self._depth
                self._dataset[start:start + data.shape[0]] = data
            self._blocks.clear()
            self._file.close()

    def __enter__(self) -> "StreamingVolumeWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
