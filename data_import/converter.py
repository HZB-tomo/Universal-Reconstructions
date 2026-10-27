"""
Converts a project read by the import plugins (e.g. a TIFF folder) into
one HDF5 file, optionally binning projections, flats and darks.

    convert_to_hdf5("scan_folder", "scan.h5", bin_y=4, bin_x=4)

The source is identified by the registered plugins exactly as in
DataImporter.import_project(); pass `plugin=` to choose one. Frames are
read lazily and written in batches, so the project never has to fit in
RAM.

Output: a projection file of the URCT HDF5 standard (doc/FILE_STANDARD.md,
paths in core/urct_format.py), read back by the `urct_hdf5` plugin:

    /entry/instrument/detector/data          projections (chunks of one frame, not compressed)
    /entry/instrument/detector/flat_fields   flat fields (if the source has them)
    /entry/instrument/detector/dark_fields   dark fields (if the source has them)
    /entry/geometry, /entry/sample, ...      every known metadata field the source has,
                                             pixel sizes multiplied by the bin factor
    /entry/provenance, /entry/software       where it came from, binning, software version

A field the source does not know is left out, not invented.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np

from core.data_interface import DataInterface
from data_import.importer import DataImporter
from data_import.plugin import ImportPlugin
from core import urct_format as fmt
from preprocessing.binning import bin_frames, binned_shape, cast_to_dtype, resolve_dtype

_PIXEL_SIZED = ("voxel_size", "detector_pixel_size")   # scaled by the horizontal bin factor
_NOT_COPIED = ("angles", "geometry_type", "center_of_rotation", "tilt_angle")  # handled separately / computed later

# progress(frames_done, frames_total, label)
ProgressFn = Callable[[int, int, str], None]


class ConversionError(Exception):
    """The source cannot be converted."""


class ConversionCancelled(Exception):
    """cancel() returned True; the partial output file was removed."""


def _as_stack(array: np.ndarray, frame_shape: tuple) -> np.ndarray:
    """Normalize a flat/dark read to (n, H, W): handles a single image or a multi-page file."""
    array = np.asarray(array)
    return array.reshape(-1, *frame_shape)


def _json_safe(metadata: dict) -> dict:
    out = {}
    for key, value in metadata.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        elif isinstance(value, (list, tuple, np.ndarray)) and len(value) <= 16:
            out[key] = np.asarray(value).tolist()
    return out


def convert_to_hdf5(
    source,
    output,
    format: Literal["tiff", "hdf5"] = "tiff",
    plugin: Optional[ImportPlugin] = None,
    bin_y: int = 1,
    bin_x: int = 1,
    method: Literal["mean", "sum"] = "mean",
    dtype: str = "auto",
    compression: Literal["gzip", "lzf", "none"] = "none",
    compression_level: int = 4,
    batch: int = 16,
    overwrite: bool = False,
    progress: Optional[ProgressFn] = None,
    cancel: Optional[Callable[[], bool]] = None,
) -> Path:
    """
    Convert `source` (a TIFF project folder, or an HDF5 file) to a
    single HDF5 file at `output` and return its path.

    Raises ConversionError if the source is not parallel-beam, has no
    projections, or has angles that do not match the projection count.
    Raises ConversionCancelled if `cancel()` returns True between batches.
    Any failure removes the partial output file.
    """
    import h5py

    if bin_y < 1 or bin_x < 1:
        raise ConversionError("bin factors must be >= 1")
    output = Path(output)
    if output.exists() and not overwrite:
        raise ConversionError(f"{output} already exists")

    project = DataImporter().import_project(source, format=format, storage_mode="disk", plugin=plugin)
    try:
        di = DataInterface(project)
        metadata = project.metadata
        geometry = metadata.get("geometry_type")
        if geometry not in (None, "parallel"):
            raise ConversionError(f"only parallel-beam projects can be converted (source is {geometry!r})")

        projections = project.data_handles["projections"]
        n_proj, height, width = projections.shape
        hb, wb = binned_shape(height, width, bin_y, bin_x)
        if hb == 0 or wb == 0:
            raise ConversionError(f"bin factor ({bin_y}, {bin_x}) is larger than the image ({height}, {width})")

        flats = darks = None
        for name in ("flats", "darks"):
            handle = project.data_handles.get(name)
            if handle is not None:
                stack = _as_stack(handle.get(None), (height, width))
                if name == "flats":
                    flats = stack
                else:
                    darks = stack

        angles = metadata.get("angles")
        if angles is not None:
            angles = np.asarray(angles, dtype=np.float64).ravel()
            if len(angles) != n_proj:
                raise ConversionError(f"source has {len(angles)} angles for {n_proj} projections")

        # Every known field the source has, with where its value came from. Unknown stays absent.
        fields = {}
        for key in fmt.FIELD_PATHS:
            if key in _NOT_COPIED:
                continue
            spec = metadata.get_field(key)
            if spec is None or spec.is_missing:
                continue
            value = spec.value
            if key in _PIXEL_SIZED:
                value = float(value) * bin_x
            fields[key] = (value, spec.source or "file")
        fields["geometry_type"] = ("parallel", "file" if geometry == "parallel" else "default")
        if angles is not None:
            fields["angles"] = (angles, "file")
        if "detector_pixel_size" not in fields and "voxel_size" in fields:
            fields["detector_pixel_size"] = (fields["voxel_size"][0], "derived")   # parallel beam: the same size
        complete = fmt.is_complete({key: value for key, (value, _source) in fields.items()})

        out_dtype = resolve_dtype(projections.get(0).dtype, method, dtype)
        chunks = (1, hb, wb)
        comp = {} if compression == "none" else {"compression": compression}
        if compression == "gzip":
            comp["compression_opts"] = compression_level

        total = n_proj + (len(flats) if flats is not None else 0) + (len(darks) if darks is not None else 0)
        done = 0

        def bin_cast(block):
            out = block if (bin_y == 1 and bin_x == 1) else bin_frames(block, bin_y, bin_x, method)
            return cast_to_dtype(out, out_dtype)

        def check_cancel():
            if cancel is not None and cancel():
                raise ConversionCancelled()

        try:
            with h5py.File(output, "w") as f:
                fmt.write_header(f, fmt.FILE_PROJECTIONS, complete=complete)
                provenance = f.require_group(fmt.PROVENANCE).attrs
                provenance["source_plugin"] = project.vendor_plugin_name or ""
                provenance["source_path"] = str(Path(source).resolve())
                provenance["source_format"] = format
                provenance["bin_y"], provenance["bin_x"] = bin_y, bin_x
                provenance["bin_method"] = method
                provenance["original_image_shape"] = (height, width)
                provenance["source_metadata"] = json.dumps(_json_safe(project.export_metadata()))
                fmt.write_software(f, "converted_by")

                ds = f.create_dataset(fmt.PROJECTIONS, shape=(n_proj, hb, wb), dtype=out_dtype, chunks=chunks, **comp)
                for start in range(0, n_proj, batch):
                    check_cancel()
                    stop = min(start + batch, n_proj)
                    ds[start:stop] = bin_cast(projections.get(slice(start, stop)))
                    if start == 0:
                        ds.attrs["axes"] = "angle,row,column"
                        ds.attrs["image_type"] = "raw"
                    done = stop
                    if progress:
                        progress(done, total, "projections")

                for name, stack, path in (("flats", flats, fmt.FLAT_FIELDS), ("darks", darks, fmt.DARK_FIELDS)):
                    if stack is None:
                        continue
                    check_cancel()
                    f.create_dataset(path, data=bin_cast(stack), chunks=chunks, **comp)
                    done += len(stack)
                    if progress:
                        progress(done, total, name)

                for key, (value, field_source) in fields.items():
                    fmt.write_field(f, key, value, field_source)
        except BaseException:
            output.unlink(missing_ok=True)  # never leave a half-written file that looks valid
            raise
    finally:
        project.close()
    return output
