"""DESY parallel-beam TIFF scan-folder convention.

Images end in _img.tif[f], _ref.tif[f], or _dar.tif[f]. A single scan log
supplies metadata; a single matching HDF5 sidecar identifies the beamline.
This export profile interprets n_angle as the number of angular positions and
ref_count as the number of projection files. It assumes one image per TIFF.
All references are passed to the backend, including both flat batches.

The registry uses micrometres for pixel/voxel sizes. Bare eff_pix values use
millimetres for this export profile; an explicit unit overrides that convention.
get_res_um is retained separately; spatial resolution is not assumed to be
pixel spacing. No camera magnification or binning correction is guessed.
"""
from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

from core.metadata import MetadataBundle
from data_import.path_map import PathMap
from data_import.plugin import ImportPlugin, register_plugin
from data_import.signature import SignatureResult, TiffSignature

_PROJECTIONS = "*_img.tif*"
_FLATS = "*_ref.tif*"
_DARKS = "*_dar.tif*"
_BEAMLINE = "/entry/beamline/name"
_HDF5_EXTENSIONS = {".h5", ".hdf5", ".nxs", ".nexus"}
_BEAMLINE_PATTERN = re.compile(r"P\d+(?=$|[\s._-])")
_START_SCAN = re.compile(r"\bstart\s+scan\b", re.IGNORECASE)
_ASSIGNMENT = re.compile(r"^\s*([A-Za-z][\w ]*?)\s*[:=]\s*(.*?)\s*$")
_QUANTITY = re.compile(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*?)")
_LENGTH_TO_UM = {"um": 1.0, "mm": 1000.0, "nm": 0.001, "m": 1_000_000.0}

# Bare-length convention inferred for the supplied log profile; registry unit is um.
EFF_PIX_UNIT: str | None = "mm"


class DesyMetadataWarning(UserWarning):
    """Metadata needs review but does not prevent reading the images."""


def _image_files(folder: Path, pattern: str) -> list[Path]:
    files = sorted(folder.glob(pattern))
    for path in files:
        if not path.is_file() or path.suffix not in {".tif", ".tiff"}:
            raise ValueError(f"DESY invalid TIFF match for {pattern!r}: {path.name}")
    if len({p.stem for p in files}) != len(files):
        raise ValueError(f"DESY duplicate image stems with different TIFF extensions: {pattern}")
    return files


def _beamline_name(path: Path) -> str | None:
    import h5py

    with h5py.File(path, "r") as handle:
        node = handle.get(_BEAMLINE)
        if not isinstance(node, h5py.Dataset) or node.size != 1:
            return None
        if h5py.check_string_dtype(node.dtype) is None:
            return None
        value = node.asstr()[()]
        if hasattr(value, "item"):
            value = value.item()
        name = value.strip().strip("\x00").strip()
        return name if _BEAMLINE_PATTERN.match(name) else None


def _read_log(path: Path) -> tuple[str, dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    starts = [i for i, line in enumerate(text.splitlines()) if _START_SCAN.search(line)]
    if len(starts) != 1:
        raise ValueError(f"DESY {path.name}: expected one Start Scan section, found {len(starts)}")
    fields: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines()[starts[0] + 1:], starts[0] + 2):
        match = _ASSIGNMENT.fullmatch(line)
        if match is None:
            continue
        key = "_".join(match[1].casefold().split())
        value = match[2].strip()
        if key in fields and fields[key] != value:
            raise ValueError(f"DESY {path.name}:{line_number}: conflicting values for {key!r}")
        fields[key] = value
    if not fields:
        raise ValueError(f"DESY {path.name}: no metadata assignments after Start Scan")
    return text, fields


def _sidecars(folder: Path) -> tuple[Path, str, Path]:
    paths = sorted(p for p in folder.iterdir() if p.is_file())
    matches, errors = [], []
    for path in paths:
        if path.suffix.lower() not in _HDF5_EXTENSIONS:
            continue
        try:
            name = _beamline_name(path)
        except (OSError, ValueError, UnicodeError, KeyError) as error:
            errors.append(f"{path.name}: {error}")
            continue
        if name is not None:
            matches.append((path, name))
    if len(matches) != 1:
        detail = "; ".join(errors)
        raise ValueError(
            f"DESY requires one HDF5 sidecar with {_BEAMLINE} matching P<number>; "
            f"found {len(matches)}." + (f" {detail}" if detail else "")
        )
    logs = []
    errors = []
    for path in paths:
        if path.suffix.lower() != ".log":
            continue
        try:
            _read_log(path)
        except (OSError, ValueError, UnicodeError) as error:
            errors.append(str(error))
            continue
        logs.append(path)
    if len(logs) != 1:
        detail = "; ".join(errors)
        raise ValueError(
            f"DESY requires one readable scan .log; found {len(logs)}."
            + (f" {detail}" if detail else "")
        )
    return matches[0][0], matches[0][1], logs[0]


def _quantity(fields: dict[str, str], key: str) -> tuple[float, str] | None:
    raw = fields.get(key, "").strip()
    if not raw:
        return None
    match = _QUANTITY.fullmatch(raw)
    if match is None:
        raise ValueError(f"DESY invalid numeric {key}: {raw!r}")
    value = float(match[1])
    if not math.isfinite(value):
        raise ValueError(f"DESY non-finite {key}: {raw!r}")
    unit = match[2].strip().replace("µ", "u").replace("μ", "u").casefold()
    return value, unit


def _count(fields: dict[str, str], key: str) -> int | None:
    quantity = _quantity(fields, key)
    if quantity is None:
        return None
    value, unit = quantity
    if unit not in {"", "frames", "images"} or value < 0 or not value.is_integer():
        raise ValueError(f"DESY {key} must be a non-negative integer, got {fields[key]!r}")
    return int(value)


def _natural_key(path: Path):
    return tuple((1, int(part)) if part.isdigit() else (0, part)
                 for part in re.split(r"(\d+)", path.name))


class DesyTiffSignature(TiffSignature):
    """Identify this folder convention, not every experiment at DESY."""

    @property
    def specificity(self) -> int:
        return 5  # Three image roles, beamline marker, and scan log.

    def check(self, folder) -> SignatureResult:
        folder = Path(folder)
        checked, failed = [], []
        try:
            for pattern in (_PROJECTIONS, _FLATS, _DARKS):
                checked.append(f"scan-root TIFFs: {pattern}")
                if not _image_files(folder, pattern):
                    failed.append(f"no files match {pattern}")
            checked.append("one P-number beamline HDF5 sidecar and one scan log")
            _sidecars(folder)
        except (OSError, ValueError) as error:
            failed.append(str(error))
        return SignatureResult(not failed, checked, failed)


@register_plugin
class DesyTiffPlugin(ImportPlugin):
    name = "desy"
    format = "tiff"
    signature = DesyTiffSignature()

    def get_path_map(self, handle) -> PathMap:
        folder = Path(handle)
        hdf5, beamline, log = _sidecars(folder)
        return PathMap(
            projections=_PROJECTIONS, flats=_FLATS, darks=_DARKS,
            metadata_source=log.name,
            extra={"hdf5_sidecar": hdf5.name, "beamline_name": beamline},
        )

    def parse_metadata(self, handle, path_map: PathMap) -> MetadataBundle:
        folder = Path(handle)
        if not path_map.metadata_source:
            raise ValueError("DESY metadata_source must identify the scan log")
        raw_log, fields = _read_log(folder / path_map.metadata_source)
        bundle = MetadataBundle()
        notes: list[str] = []
        counts = {}
        for key in ("projections", "flats", "darks"):
            pattern = getattr(path_map, key)
            if not pattern:
                raise ValueError(f"DESY missing {key} pattern")
            files = _image_files(folder, pattern)
            counts[key] = len(files)
            if files:
                bundle.set(f"{key}_path", pattern, tier="mandatory_hard")
            else:
                bundle.declare_missing(f"{key}_path", tier="mandatory_hard")
            if key == "projections" and files != sorted(files, key=_natural_key):
                raise ValueError(
                    "DESY projection filenames do not sort in numeric order. "
                    "The TIFF backend sorts lexically; use zero-padded sequence numbers."
                )
        bundle.set("projection_count", counts["projections"], tier="optional", source="computed")
        bundle.set("geometry_type", "parallel", tier="mandatory_soft", source="computed")
        for key in ("angles", "angle_span", "voxel_size", "detector_pixel_size"):
            bundle.declare_missing(key, tier="mandatory_soft")
        for key in ("center_of_rotation", "tilt_angle"):
            bundle.declare_missing(key, tier="computed")
        for key in ("energy", "acquisition_datetime"):
            bundle.declare_missing(key, tier="optional")

        bundle.set("desy_log_text", raw_log, tier="optional")
        bundle.set("desy_log_fields", fields, tier="optional")
        bundle.set("desy_image_counts", counts, tier="optional", source="computed")
        bundle.set("desy_flat_reference_policy", "all reference frames pooled", tier="optional", source="computed")
        if path_map.extra.get("beamline_name"):
            bundle.set("instrument_name", path_map.extra["beamline_name"], tier="optional")
        if path_map.extra.get("hdf5_sidecar"):
            bundle.set("desy_hdf5_sidecar", path_map.extra["hdf5_sidecar"], tier="optional")

        declared = {key: _count(fields, key) for key in ("n_dark", "n_ref", "n_img", "n_angle", "ref_count")}
        if declared["n_angle"] is not None:
            if declared["n_angle"] <= 0:
                raise ValueError("DESY n_angle must be a positive count of angular positions")
            bundle.set("desy_angle_count", declared["n_angle"], tier="optional")
            if declared["n_img"] == 1 and declared["n_angle"] != counts["projections"]:
                raise ValueError(
                    f"DESY n_angle={declared['n_angle']} with n_img=1, "
                    f"but found {counts['projections']} projection files."
                )
        if declared["ref_count"] is not None:
            expected = declared["ref_count"]
            if expected <= 0 or expected != counts["projections"]:
                raise ValueError(
                    f"DESY ref_count declares {expected} projections but found {counts['projections']} "
                    "*_img TIFFs. Check completeness and this export's ref_count meaning."
                )
        if declared["n_dark"] is not None and declared["n_dark"] != counts["darks"]:
            notes.append(f"n_dark={declared['n_dark']}, actual dark files={counts['darks']}")
        if declared["n_ref"] is not None and declared["n_ref"] != counts["flats"]:
            notes.append(
                f"n_ref={declared['n_ref']}, actual reference files={counts['flats']}; "
                "n_ref may describe one batch. All reference files are included."
            )

        # Angular counts do not specify scan coverage. Only read an explicit span.
        span = _quantity(fields, "angle_span")
        if span is not None:
            value, unit = span
            if value <= 0 or unit not in {"", "deg", "degree", "degrees", "°"}:
                raise ValueError("DESY angle_span must be positive and expressed in degrees")
            bundle.set("angle_span", value, tier="mandatory_soft")

        pixel = _quantity(fields, "eff_pix")
        if pixel is not None:
            value, unit = pixel
            if value <= 0:
                raise ValueError("DESY eff_pix must be positive")
            unit = unit or EFF_PIX_UNIT
            if unit is None:
                notes.append("eff_pix has no unit; pixel/voxel size remains unresolved. Confirm EFF_PIX_UNIT.")
            else:
                unit = unit.replace("µ", "u").replace("μ", "u").casefold().strip()
                if unit not in _LENGTH_TO_UM:
                    raise ValueError(f"DESY unsupported eff_pix unit: {unit!r}")
                pixel_um = value * _LENGTH_TO_UM[unit]
                if not math.isfinite(pixel_um) or pixel_um <= 0:
                    raise ValueError("DESY eff_pix cannot be represented in micrometres")
                bundle.set("detector_pixel_size", pixel_um, tier="mandatory_soft")
                bundle.set("voxel_size", pixel_um, tier="mandatory_soft")

        exposure = _quantity(fields, "exptime")
        if exposure is not None:
            value, unit = exposure
            if value < 0:
                raise ValueError("DESY exptime must be non-negative")
            if unit in {"s", "ms", "us", "sec", "seconds"}:
                factor = {"ms": 0.001, "us": 0.000001}.get(unit, 1.0)
                bundle.set("exposure_time_s", value * factor, tier="optional")
            elif unit:
                notes.append(f"exptime unit {unit!r} is not recognized; retained in desy_log_fields")

        bundle.set("desy_metadata_notes", notes, tier="optional", source="computed")
        for note in notes:
            warnings.warn(note, DesyMetadataWarning, stacklevel=2)
        return bundle
