"""
The volume file of the URCT HDF5 standard (doc/FILE_STANDARD.md): a reconstructed
volume, how it was made, and the measurement it came from. Used by save_volume() and
StreamingVolumeWriter (export.py); read back with read_volume_info().

    /entry/volume/data            (z, y, x) float32 or uint16, cubic-ish chunks, not compressed
    /entry/reconstruction         the parameters that shaped the volume (attributes; a dict is a subgroup)
    /entry/reconstruction/preprocessing
    /entry/geometry, /entry/sample, /entry/instrument, ...   the measurement metadata, same paths as in a projection file
    /entry/source                 uuid / path of the projection file it came from
    /entry/software               who wrote it

Everything is derived from the one flat `settings` dict callers already build
(ProjectObject.export_metadata() merged with the run's settings): known metadata
fields go to the measurement paths, the keys named in RECORD_KEYS to the
reconstruction record, and the whole dict is still kept as JSON in the dataset's
`reconstruction_settings` attribute, so nothing a caller passes is lost.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Optional

import numpy as np

from core import urct_format as fmt

CHUNK_EDGE = 64                      # decided by benchmarks/bench_volume_chunks.py, see the standard
BUFFER_LIMIT = 2 << 30               # bytes: the writer buffers one chunk-deep block of slices up to this size
RECORD_GROUP = "/entry/reconstruction"
SOURCE_GROUP = "/entry/source"

# settings key -> name in /entry/reconstruction
RECORD_KEYS = {
    "backend": "algorithm",
    "filter_name": "filter",
    "num_iter": "num_iter",
    "algorithm_options": "algorithm_options",
    "device_requested": "device_requested",
    "device_used": "device_used",
    "center": "center_of_rotation",
    "tilt_deg": "tilt_angle",
    "rotate_deg": "rotation_offset",
    "rows": "rows",
    "crop": "crop",
    "n_slices": "n_slices",
    "duration_s": "duration_s",
    "speed_slices_per_s": "speed_slices_per_s",
    "started": "started",
    "finished": "finished",
    "calibration_backend": "calibration_backend",
    "dtype": "dtype",
}
LIBRARIES = ("algotom", "numba", "urct_gridrec", "itk-rtk", "tomopy", "astra-toolbox", "cupy", "numpy", "h5py")


def chunk_shape(shape: tuple, dtype) -> tuple:
    """Cubic chunks of about 1 MiB: 64^3 for float32, 64 x 64 x 128 for uint16."""
    n, h, w = shape
    x_edge = 2 * CHUNK_EDGE if np.dtype(dtype).itemsize == 2 else CHUNK_EDGE
    return (max(1, min(CHUNK_EDGE, n)), max(1, min(CHUNK_EDGE, h)), max(1, min(x_edge, w)))


# -- writing -----------------------------------------------------------------

def _put(group, name: str, value: Any) -> None:
    """One record entry: dict -> subgroup, short arrays/scalars/strings -> attributes, long arrays -> datasets."""
    if value is None:
        return
    if isinstance(value, dict):
        sub = group.require_group(name)
        for key, item in value.items():
            _put(sub, str(key), item)
        return
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            value = np.asarray(value)
        except ValueError:
            value = json.dumps(value, default=str)
        else:
            if value.dtype == object:
                value = json.dumps(np.asarray(value).tolist(), default=str)
            elif value.size > 64:
                group.create_dataset(name, data=value)
                return
    group.attrs[name] = value


def _put_clip(group, clip: dict) -> None:
    for key in ("clip_min", "clip_max", "source_dtype"):
        _put(group, key, clip.get(key))


def library_versions() -> dict:
    out = {}
    for name in LIBRARIES:
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return out


def units_and_scale(settings: dict, dtype) -> dict:
    """The dataset attributes that say what its numbers mean."""
    attrs = {"axes": "z,y,x"}
    clip = settings.get("dtype_clip")
    if np.dtype(dtype) == np.uint16 and clip:
        lo, hi = float(clip["clip_min"]), float(clip["clip_max"])
        attrs.update(scale=(hi - lo) / 65535.0, offset=lo)
        attrs["units"] = "1/cm" if settings.get("detector_pixel_size") else "1/px"
        attrs["stored_as"] = "uint16: value = offset + scale * stored"
    elif np.dtype(dtype) == np.uint16:
        attrs["units"] = "uint16 counts, scale unknown"
    else:
        attrs["units"] = "1/cm" if settings.get("detector_pixel_size") else "1/px"
    voxel = settings.get("voxel_size") or settings.get("detector_pixel_size")
    if voxel:
        attrs["voxel_size"] = float(voxel)
    return attrs


def write_volume_metadata(f, dataset, settings: Optional[dict]) -> None:
    """Header, dataset attributes, reconstruction record, measurement, source and software."""
    given = bool(settings)
    settings = dict(settings or {})
    fmt.write_header(f, fmt.FILE_VOLUME, complete=fmt.is_complete(
        {key: settings.get(key) for key in ("geometry_type", "angles", "angle_span", "voxel_size", "detector_pixel_size",
                                              "source_to_detector_distance", "source_to_object_distance")}))
    for name, value in units_and_scale(settings, dataset.dtype).items():
        dataset.attrs[name] = value

    update_record(f, settings)

    for key in fmt.FIELD_PATHS:                       # the measurement, at the projection file's paths
        value = settings.get(key)
        if value is not None:
            fmt.write_field(f, key, value)
    source = f.require_group(SOURCE_GROUP).attrs
    if settings.get("source_file_uuid"):
        source["uuid"] = str(settings["source_file_uuid"])
    if settings.get("source_file"):
        source["path"] = str(settings["source_file"])
    fmt.write_software(f, "reconstructed_by", library_versions())
    if given:
        try:
            dataset.attrs["reconstruction_settings"] = json.dumps(_jsonable(settings))
        except (TypeError, ValueError):
            pass


def update_record(f, settings: dict) -> None:
    """(Re)write the reconstruction record from `settings`; entries not in `settings` are kept.
    Called at creation and again at the end of a streamed run for the timings."""
    group = f.require_group(RECORD_GROUP)
    for key, name in RECORD_KEYS.items():
        if key in settings:
            if name in group.attrs:
                del group.attrs[name]
            if name in group:
                del group[name]
            _put(group, name, settings[key])
    if isinstance(settings.get("dtype_clip"), dict):
        _put_clip(group, settings["dtype_clip"])
    if isinstance(settings.get("preprocessing"), dict):
        if "preprocessing" in group:
            del group["preprocessing"]
        _put(group, "preprocessing", settings["preprocessing"])


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def create_volume_file(path, shape: tuple, dtype, settings: Optional[dict] = None, dataset_path: str = fmt.VOLUME):
    """Open `path` for writing and create the chunked, uncompressed volume dataset with
    all metadata. Returns (h5py.File, dataset); the caller fills the dataset and closes the file."""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    try:
        dataset = f.create_dataset(dataset_path, shape=shape, dtype=dtype, chunks=chunk_shape(shape, dtype))
        write_volume_metadata(f, dataset, settings)
    except BaseException:
        f.close()
        raise
    return f, dataset


# -- reading -----------------------------------------------------------------

@dataclass
class VolumeInfo:
    path: str
    dataset_path: str
    shape: tuple
    dtype: str
    chunks: Optional[tuple]
    units: str
    voxel_size: Optional[float]               # um
    scale: Optional[float]                    # uint16 volumes: value = offset + scale * stored
    offset: Optional[float]
    uuid: Optional[str]
    version: Optional[str]
    complete: bool
    record: dict = field(default_factory=dict)          # /entry/reconstruction, nested like the file
    measurement: dict = field(default_factory=dict)     # known metadata field -> value
    source: dict = field(default_factory=dict)          # uuid / path of the projection file
    software: dict = field(default_factory=dict)

    def to_physical(self, stored: np.ndarray) -> np.ndarray:
        """Stored values as attenuation coefficients (float32): undoes the uint16 scaling if there is any."""
        if self.scale is not None and self.offset is not None:
            return (np.asarray(stored, dtype=np.float32) * np.float32(self.scale) + np.float32(self.offset)).astype(np.float32)
        return np.asarray(stored, dtype=np.float32)


def find_volume_dataset(f) -> Optional[str]:
    """The standard dataset, else the legacy "/volume"; None if neither exists."""
    for path in (fmt.VOLUME, "/volume"):
        if path in f:
            return path
    return None


def _read_group(group) -> dict:
    out = {}
    for name, value in group.attrs.items():
        out[name] = value.item() if isinstance(value, np.generic) else (value.decode() if isinstance(value, bytes) else value)
    for name, node in group.items():
        out[name] = _read_group(node) if hasattr(node, "items") else fmt.read_value(node)
    return out


def read_volume_info(path, dataset_path: Optional[str] = None) -> VolumeInfo:
    """Everything about a volume file except the voxels. Reads the legacy layout (a bare
    `/volume` dataset with a reconstruction_settings attribute) as well."""
    import h5py

    with h5py.File(path, "r") as f:
        dataset_path = dataset_path or find_volume_dataset(f)
        if dataset_path is None or dataset_path not in f:
            raise ValueError(f"{path} has no volume dataset ({fmt.VOLUME} or /volume)")
        ds = f[dataset_path]
        attrs = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in ds.attrs.items()}
        standard = f.attrs.get("urct_standard") == fmt.STANDARD
        info = VolumeInfo(
            path=str(path), dataset_path=dataset_path, shape=tuple(ds.shape), dtype=str(ds.dtype), chunks=ds.chunks,
            units=str(attrs.get("units", "unknown")), voxel_size=attrs.get("voxel_size"),
            scale=attrs.get("scale"), offset=attrs.get("offset"),
            uuid=f.attrs.get("urct_uuid"), version=f.attrs.get("urct_version"),
            complete=bool(f.attrs.get("urct_complete", False)),
        )
        if standard:
            if RECORD_GROUP in f:
                info.record = _read_group(f[RECORD_GROUP])
            info.measurement = {key: found[0] for key in fmt.FIELD_PATHS if (found := fmt.find_field(f, key)) is not None}
            if SOURCE_GROUP in f:
                info.source = _read_group(f[SOURCE_GROUP])
            if fmt.SOFTWARE in f:
                info.software = _read_group(f[fmt.SOFTWARE])
        elif "reconstruction_settings" in attrs:
            try:
                info.record = json.loads(attrs["reconstruction_settings"])
            except ValueError:
                pass
    return info
