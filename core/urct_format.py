"""
The URCT HDF5 file standard: paths, header attributes, and helpers shared by
everything that reads or writes it (the `urct_hdf5` import plugin, the
converter, and later the volume writer). See doc/FILE_STANDARD.md for the
reasoning; this file is the machine-readable copy of its tables.

Two file kinds, never mixed: "projections" and "volume". Both carry the same
metadata groups at the same paths (/entry/geometry, /entry/sample, ...).
"""
from __future__ import annotations

import platform
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from core.version import __version__

STANDARD = "urct-hdf5"
FORMAT_VERSION = (0, 1)          # (major, minor) of the file standard, not of the software
FILE_PROJECTIONS = "projections"
FILE_VOLUME = "volume"

# -- datasets ----------------------------------------------------------------
PROJECTIONS = "/entry/instrument/detector/data"
FLAT_FIELDS = "/entry/instrument/detector/flat_fields"
DARK_FIELDS = "/entry/instrument/detector/dark_fields"
VOLUME = "/entry/volume/data"
SOFTWARE = "/entry/software"
PROVENANCE = "/entry/provenance"
EXTRA = "/entry/extra"

# Metadata field (core/known_fields.py key) -> dataset path.
FIELD_PATHS: dict[str, str] = {
    "angles": "/entry/sample/rotation_angle",
    "geometry_type": "/entry/geometry/geometry_type",
    "angle_span": "/entry/geometry/angle_span",
    "rotation_direction": "/entry/geometry/rotation_direction",
    "voxel_size": "/entry/geometry/voxel_size",
    "detector_pixel_size": "/entry/geometry/pixel_size",
    "source_to_detector_distance": "/entry/geometry/source_to_detector_distance",
    "source_to_object_distance": "/entry/geometry/source_to_object_distance",
    "propagation_distance": "/entry/geometry/propagation_distance",
    "center_of_rotation": "/entry/geometry/center_of_rotation",
    "tilt_angle": "/entry/geometry/tilt_angle",
    "energy": "/entry/instrument/source/energy",
    "operator": "/entry/operator",
    "instrument_name": "/entry/instrument/name",
    "sample_name": "/entry/sample/name",
    "acquisition_datetime": "/entry/start_time",
}

# Older locations that are still accepted when reading (field -> paths, tried in order).
LEGACY_PATHS: dict[str, tuple] = {
    "angles": ("/entry/instrument/sample/rotation_angle",),
}

UNITS = {
    "angles": "degrees", "angle_span": "degrees", "tilt_angle": "degrees",
    "voxel_size": "um", "detector_pixel_size": "um",
    "source_to_detector_distance": "mm", "source_to_object_distance": "mm", "propagation_distance": "mm",
    "center_of_rotation": "pixels", "energy": "keV",
}


# -- header ------------------------------------------------------------------

def write_header(f, file_type: str, complete: Optional[bool] = None) -> None:
    """The root attributes every file of the standard has."""
    f.attrs["urct_standard"] = STANDARD
    f.attrs["urct_version"] = f"{FORMAT_VERSION[0]}.{FORMAT_VERSION[1]}"
    f.attrs["urct_file_type"] = file_type
    f.attrs["urct_uuid"] = str(uuid.uuid4())
    if complete is not None:
        f.attrs["urct_complete"] = bool(complete)


def parse_version(text: str) -> tuple:
    major, _, minor = str(text).partition(".")
    return int(major), int(minor or 0)


def software_record(libraries: Optional[dict] = None) -> dict:
    """Who wrote the file: the software version, the git commit when the code
    runs from a checkout, Python, and any library versions the caller names."""
    record = {"urct": __version__, "python": platform.python_version(), "written": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    commit = _git_commit()
    if commit:
        record["git_commit"] = commit
    for name, version in (libraries or {}).items():
        record[name] = str(version)
    return record


def _git_commit() -> Optional[str]:
    try:
        root = Path(__file__).resolve().parents[1]
        if not (root / ".git").exists():
            return None
        out = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=root, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def write_software(f, role: str, libraries: Optional[dict] = None) -> None:
    """/entry/software attributes; `role` is "converted_by" or "reconstructed_by"."""
    group = f.require_group(SOFTWARE)
    group.attrs["role"] = role
    for key, value in software_record(libraries).items():
        group.attrs[key] = value


# -- metadata fields ---------------------------------------------------------

def write_field(f, key: str, value: Any, source: str = "file") -> bool:
    """Write one known metadata field at its standard path. Returns False for
    an unknown key or a missing (None) value: unknown is absent, never made up."""
    path = FIELD_PATHS.get(key)
    if path is None or value is None:
        return False
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, str):
        import h5py

        dataset = f.create_dataset(path, data=value, dtype=h5py.string_dtype("utf-8"))
    else:
        dataset = f.create_dataset(path, data=value)
    if key in UNITS:
        dataset.attrs["units"] = UNITS[key]
    if source and source != "file":
        dataset.attrs["source"] = source
    return True


def read_value(dataset):
    """A dataset's value as plain Python/numpy: strings decoded, scalars unboxed."""
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, np.generic):
        value = value.item()
        return value.decode("utf-8") if isinstance(value, bytes) else value
    return value


def find_field(f, key: str):
    """(value, source, path) of a field, looking at the standard path and then the
    legacy ones; None if absent."""
    for path in (FIELD_PATHS.get(key), *LEGACY_PATHS.get(key, ())):
        if path and path in f:
            dataset = f[path]
            source = dataset.attrs.get("source", "file")
            if isinstance(source, bytes):
                source = source.decode()
            return read_value(dataset), str(source), path
    return None


def is_complete(values: dict) -> bool:
    """Whether the mandatory_soft fields are all there: `values` maps field -> value
    (None = missing). Angles may be replaced by angle_span, and pixel size by
    voxel size (parallel beam: the same number); cone beam also needs both distances."""
    has = lambda key: values.get(key) is not None
    if not has("geometry_type"):
        return False
    if not (has("angles") or has("angle_span")):
        return False
    if not (has("voxel_size") or has("detector_pixel_size")):
        return False
    if values.get("geometry_type") == "cone":
        return has("source_to_detector_distance") and has("source_to_object_distance")
    return True


# -- checking a file ---------------------------------------------------------

@dataclass
class FileReport:
    file_type: Optional[str] = None
    version: Optional[str] = None
    errors: list = field(default_factory=list)       # the file is not usable
    missing: list = field(default_factory=list)      # mandatory_soft fields not in the file
    complete: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors


def check_file(path) -> FileReport:
    """Validate a file against the standard: header, required datasets, and which
    soft fields are missing. Never raises for a bad file; the problems are in the report."""
    import h5py

    report = FileReport()
    try:
        with h5py.File(path, "r") as f:
            if f.attrs.get("urct_standard") != STANDARD:
                report.errors.append("not a URCT HDF5 file (root attribute urct_standard is missing)")
                return report
            report.version = str(f.attrs.get("urct_version", ""))
            report.file_type = str(f.attrs.get("urct_file_type", ""))
            try:
                major, _minor = parse_version(report.version)
            except ValueError:
                report.errors.append(f"unreadable urct_version {report.version!r}")
                return report
            if major > FORMAT_VERSION[0]:
                report.errors.append(f"file standard {report.version} is newer than this software understands ({FORMAT_VERSION[0]}.{FORMAT_VERSION[1]})")
            required = {FILE_PROJECTIONS: PROJECTIONS, FILE_VOLUME: VOLUME}.get(report.file_type)
            if required is None:
                report.errors.append(f"unknown urct_file_type {report.file_type!r}")
            elif required not in f:
                report.errors.append(f"required dataset {required} is missing")
            if "/entry/geometry/geometry_type" not in f:
                report.errors.append("required field geometry_type is missing")
            values = {key: (find_field(f, key) or (None,))[0] for key in FIELD_PATHS}
            report.complete = is_complete(values)
            if values.get("geometry_type") is not None:
                for key, label in (("angles", "angles or angle_span"), ("voxel_size", "voxel_size or pixel_size")):
                    alt = {"angles": "angle_span", "voxel_size": "detector_pixel_size"}[key]
                    if values.get(key) is None and values.get(alt) is None:
                        report.missing.append(label)
                if values.get("geometry_type") == "cone":
                    report.missing += [k for k in ("source_to_detector_distance", "source_to_object_distance") if values.get(k) is None]
    except OSError as e:
        report.errors.append(f"cannot open the file: {e}")
    return report
