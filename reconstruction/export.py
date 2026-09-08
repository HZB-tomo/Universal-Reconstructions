"""
save_volume(): writes a reconstructed volume to disk, as a TIFF stack
or a single HDF5 dataset, with an optional JSON settings/provenance
record (backend used, center, tilt, ring correction, crop, dtype
clipping, ...; see core/project.py's ProjectObject.export_metadata()
for pulling acquisition metadata into the same dict).
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
