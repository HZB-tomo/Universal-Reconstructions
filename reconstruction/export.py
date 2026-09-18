"""
save_volume(): writes a reconstructed volume to disk, as a TIFF stack
or a single HDF5 dataset, with an optional JSON settings/provenance
record (backend used, center, tilt, ring correction, crop, dtype
clipping, ...; see core/project.py's ProjectObject.export_metadata()
for pulling acquisition metadata into the same dict).

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


def save_volume(
    volume: np.ndarray,
    path: Union[str, Path],
    format: Literal["tiff", "hdf5"],
    dataset_path: str = "/volume",
    settings: Optional[dict] = None,
) -> None:
    """
    Write a reconstructed volume to disk. `volume` is a (n_slices,
    height, width) stack, or a single (height, width) image.

    format="tiff" writes one file per slice into the folder at `path`
    (created if it doesn't exist yet), named `slice_00000.tif`,
    `slice_00001.tif`, ... . This is the same folder-of-files layout
    data_import's TiffBackend reads.

    format="hdf5" writes a single dataset at `dataset_path` inside the
    HDF5 file at `path` (created, or overwritten if it already exists).

    `settings`, if given, is a plain dict recording anything about how
    this volume was produced: reconstruction backend and options,
    center, tilt, whether ring correction ran, crop bounds, dtype-clip
    range, or acquisition metadata from ProjectObject.export_metadata().
    Written as JSON: an HDF5 attribute (`reconstruction_settings`, on
    the dataset) for format="hdf5", or a `metadata.json` file alongside
    the slices for format="tiff". Values must be JSON-serializable;
    numpy scalars and arrays are converted automatically.

    Neither format records units on its own. See reconstruct()'s
    docstring for what a volume's values mean (1/px or 1/um, depending
    on whether geometry.detector_pixel_size was set) -- worth carrying
    in `settings` if this volume will outlive the session that made it.
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
    import h5py  # imported here so this package stays importable without h5py installed

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        dataset = f.create_dataset(dataset_path, data=volume)
        if settings:
            dataset.attrs["reconstruction_settings"] = json.dumps(_to_jsonable(settings))


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
        dataset_path: str = "/volume",
        settings: Optional[dict] = None,
    ):
        self._format = format
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
            import h5py  # imported here so this package stays importable without h5py installed

            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = h5py.File(path, "w")
            self._dataset = self._file.create_dataset(dataset_path, shape=(total_slices, height, width), dtype=dtype)
            if settings:
                self._dataset.attrs["reconstruction_settings"] = json.dumps(_to_jsonable(settings))
        else:
            raise ValueError(f"Unknown format {format!r}, expected 'tiff' or 'hdf5'")

    def write_chunk(self, chunk: np.ndarray, z_offset: int) -> None:
        """`chunk` is (n, height, width); written at absolute slice
        indices [z_offset, z_offset + n). See class docstring for why
        z_offset doesn't need to arrive in increasing order."""
        if self._format == "tiff":
            for i, image in enumerate(chunk):
                self._tifffile.imwrite(self._folder / f"slice_{z_offset + i:0{self._n_digits}d}.tif", image)
        else:
            n = chunk.shape[0]
            self._dataset[z_offset:z_offset + n] = chunk

    def close(self) -> None:
        if self._format == "hdf5":
            self._file.close()

    def __enter__(self) -> "StreamingVolumeWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
