"""RX Solution TIFF import with XML-selected geometry and complete XML metadata.

Unknown XML fields are retained without assigning them reconstruction meaning.
Distances enter the standard millimetre fields only when their unit is known.
Angular coverage defaults to an editable 360 degrees; completion derives angles.
"""
from __future__ import annotations

import base64
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from core.metadata import MetadataBundle
from data_import.path_map import PathMap
from data_import.plugin import ImportPlugin, register_plugin
from data_import.signature import SignatureResult, TiffSignature

_PROJECTIONS = "Proj/proj_*.tif*"
_FLATS = "FlatsXRays.tif*"
_DARKS = "DarksXRays.tif*"
_METADATA = "unireconstruction.xml"
_TIFF_SUFFIXES = {".tif", ".tiff"}
_LENGTH_TO_MM = {"m": 1000.0, "mm": 1.0, "um": 0.001, "nm": 0.000001}

# Set to "mm" only after confirming the unit of geometry@sod and geometry@sdd.
DISTANCE_UNIT: str | None = "mm"
DEFAULT_ANGLE_SPAN = 360.0


def _xml_records(root):
    """Retain hierarchy, repeated elements, blank attributes, text, and tails."""
    records = []

    def visit(node, path):
        records.append({
            "path": path, "element": node.tag, "attributes": dict(node.attrib),
            "text": node.text, "tail": node.tail,
        })
        occurrences = {}
        for child in node:
            occurrences[child.tag] = occurrences.get(child.tag, 0) + 1
            visit(child, f"{path}/{child.tag}[{occurrences[child.tag]}]")

    visit(root, f"/{root.tag}[1]")
    return records


def _number(node, attribute: str, *, integer=False):
    raw = node.get(attribute) if node is not None else None
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw) if integer else float(raw)
    except ValueError as error:
        raise ValueError(f"RX Solution invalid {node.tag}@{attribute}: {raw!r}") from error
    if not integer and not math.isfinite(value):
        raise ValueError(f"RX Solution non-finite {node.tag}@{attribute}: {raw!r}")
    return value


def _one(parent, *tags):
    if parent is None:
        return None
    nodes = [node for tag in tags for node in parent.findall(tag)]
    if len(nodes) > 1:
        raise ValueError(f"RX Solution duplicate XML section: {' / '.join(tags)}")
    return nodes[0] if nodes else None


def _xyz(node, *, integer=False):
    if node is None:
        return None
    values = tuple(_number(node, axis, integer=integer) for axis in "XYZ")
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"RX Solution <{node.tag}> requires all of X, Y, Z")
    return values


def _metadata(path: Path, *, xml_bytes=None):
    try:
        root = ET.fromstring(path.read_bytes() if xml_bytes is None else xml_bytes)
    except (OSError, ET.ParseError) as error:
        raise ValueError(f"RX Solution cannot read {path}: {error}") from error
    if root.tag != "unireconstruction":
        raise ValueError("RX Solution metadata root must be <unireconstruction>")
    section = _one(root, "conebeam", "parallelbeam")
    if section is None:
        raise ValueError("RX Solution requires one <conebeam> or <parallelbeam> section")
    return root, section


def _projection_files(folder: Path, pattern: str) -> list[Path]:
    files = sorted(folder.glob(pattern))
    for path in files:
        if not path.is_file() or path.suffix not in _TIFF_SUFFIXES:
            raise ValueError(f"RX Solution invalid projection match: {path.name}")
    if len({path.stem for path in files}) != len(files):
        raise ValueError("RX Solution duplicate projection stems with different TIFF extensions")
    return files


def _reference_file(folder: Path, pattern: str, label: str) -> Path:
    """Locate a flat/dark file by name only. The file is never opened or read."""
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise ValueError(f"RX Solution {label} file not found ({pattern})")
    if len(matches) > 1:
        raise ValueError(
            f"RX Solution multiple {label} files match {pattern}: {[p.name for p in matches]}"
        )
    path = matches[0]
    if not path.is_file() or path.suffix not in _TIFF_SUFFIXES:
        raise ValueError(f"RX Solution invalid {label} match: {path.name}")
    return path


def _validate_projections(files: list[Path]):
    import tifffile

    def numeric_key(path):
        return tuple((1, int(part)) if part.isdigit() else (0, part)
                     for part in re.split(r"(\d+)", path.name))

    if files != sorted(files, key=numeric_key):
        raise ValueError("RX Solution projection order differs from numeric order; use zero-padded filenames")
    expected_shape = None
    for path in files:
        with tifffile.TiffFile(path) as tiff:
            if len(tiff.pages) != 1 or len(tiff.series) != 1 or tiff.series[0].ndim != 2:
                raise ValueError(f"RX Solution expects one 2D grayscale frame per TIFF: {path}")
            series = tiff.series[0]
            if series.dtype.kind not in "buif":
                raise ValueError(f"RX Solution unsupported TIFF dtype {series.dtype}: {path}")
            if expected_shape is None:
                expected_shape = series.shape
            elif series.shape != expected_shape:
                raise ValueError(f"RX Solution projection shape {series.shape} differs from {expected_shape}: {path}")
    return expected_shape


def _unit(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("RX Solution units must be text or None")
    value = value.strip().replace("µ", "u").replace("μ", "u").casefold()
    if value not in _LENGTH_TO_MM:
        raise ValueError("RX Solution length units must be m, mm, um, nm, or None")
    return value


def _convert(value, factor, label):
    converted = value * factor
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"RX Solution {label} cannot be represented in the target unit")
    return converted


class RxSolnTiffSignature(TiffSignature):
    """Recognize the RX Solution layout: projections, flat/dark names, and XML."""

    @property
    def specificity(self):
        # Projection filenames, FlatsXRays name, DarksXRays name,
        # XML root + one supported geometry section.
        return 4

    def check(self, folder):
        folder = Path(folder)
        checked = [
            "scan-root Proj/proj_*.tif[f]",
            "scan-root FlatsXRays.tif[f]",
            "scan-root DarksXRays.tif[f]",
            "unireconstruction XML root",
            "one supported geometry section",
        ]
        failed = []
        try:
            if not _projection_files(folder, _PROJECTIONS):
                failed.append("no RX Solution projection TIFFs found")
        except (OSError, ValueError) as error:
            failed.append(str(error))
        for pattern, label in ((_FLATS, "flat"), (_DARKS, "dark")):
            try:
                _reference_file(folder, pattern, label)  # name check only
            except (OSError, ValueError) as error:
                failed.append(str(error))
        try:
            _metadata(folder / _METADATA)
        except (OSError, ValueError) as error:
            failed.append(str(error))
        return SignatureResult(not failed, checked, failed)


@register_plugin
class RxSolnTiffPlugin(ImportPlugin):
    name = "rx_soln"
    format = "tiff"
    signature = RxSolnTiffSignature()

    def __init__(self, *, voxel_unit="mm", distance_unit=DISTANCE_UNIT):
        # RX Solution XML voxelSize is assumed to be in mm unless told otherwise.
        # Pass voxel_unit=None explicitly to leave the size unresolved.
        self.voxel_unit = _unit(voxel_unit)
        self.distance_unit = _unit(distance_unit)

    def get_path_map(self, handle) -> PathMap:
        # Flats/darks are registered so the rest of the software can resolve them.
        return PathMap(
            projections=_PROJECTIONS,
            flats=_FLATS,
            darks=_DARKS,
            metadata_source=_METADATA,
        )

    def parse_metadata(self, handle, path_map: PathMap) -> MetadataBundle:
        folder = Path(handle)
        if not path_map.metadata_source or not path_map.projections:
            raise ValueError("RX Solution requires projection and XML metadata paths")
        xml_path = folder / path_map.metadata_source
        xml_bytes = xml_path.read_bytes()
        root, section = _metadata(xml_path, xml_bytes=xml_bytes)
        files = _projection_files(folder, path_map.projections)
        # Only projections are opened/validated. Flats/darks are intentionally
        # not read: no shape check, no metadata entries.
        image_shape = _validate_projections(files)
        bundle = MetadataBundle()
        if files:
            bundle.set("projections_path", path_map.projections, tier="mandatory_hard")
        else:
            bundle.declare_missing("projections_path", tier="mandatory_hard")
        bundle.set("projections_normalized", True, tier="optional", source="computed")
        bundle.set("projection_count", len(files), tier="optional", source="computed")
        if image_shape is not None:
            bundle.set("rx_sol_projection_shape", image_shape, tier="optional", source="file")

        geometry_type = "cone" if section.tag == "conebeam" else "parallel"
        bundle.set("geometry_type", geometry_type, tier="mandatory_soft")

        def optional(key, value, source="file"):
            if value is not None and (not isinstance(value, str) or value.strip()):
                bundle.set(key, value, tier="optional", source=source)

        records = _xml_records(root)
        optional("rx_sol_xml_attributes", records)
        optional("rx_sol_xml_document", ET.tostring(root, encoding="unicode"))
        # Preserve the exact original bytes too (including encoding and comments).
        optional("rx_sol_xml_original_base64", base64.b64encode(xml_bytes).decode("ascii"))
        xml_values = {}
        for record in records:
            path = record["path"]
            for attribute, value in record["attributes"].items():
                xml_values[f"{path}/@{attribute}"] = value
            if record["text"] is not None:
                xml_values[f"{path}/text()"] = record["text"]
            if record["tail"] is not None:
                xml_values[f"{path}/tail()"] = record["tail"]
        optional("rx_sol_xml_values", xml_values)
        for path, value in xml_values.items():
            # Call set directly so empty strings and whitespace are not dropped.
            bundle.set(f"rx_sol_xml_value:{path}", value, tier="optional")
        notes = []
        profile = _one(section, "profile")
        declared = _number(profile, "images", integer=True)
        if declared is not None:
            if declared <= 0 or declared != len(files):
                raise ValueError(f"RX Solution XML declares {declared} projections, but found {len(files)} files")
            optional("rx_sol_declared_images", declared)
        optional("turns", _number(profile, "turns"))
        bundle.declare_missing("angles", tier="mandatory_soft")
        bundle.set("angle_span", DEFAULT_ANGLE_SPAN, tier="mandatory_soft", source="default")
        for key in ("center_of_rotation", "tilt_angle"):
            bundle.declare_missing(key, tier="computed")

        volume = _one(section, "volume")
        voxel = _xyz(_one(volume, "voxelSize"))
        pixel_size_um = None
        if voxel is not None:
            if min(voxel) < 0:
                raise ValueError("RX Solution voxel sizes must be non-negative")
            optional("rx_sol_voxel_size_xyz", voxel)
            if self.voxel_unit is not None and min(voxel) > 0 and all(
                math.isclose(v, voxel[0], rel_tol=1e-6, abs_tol=0) for v in voxel
            ):
                # Target unit is um (known_fields default is 1.0 um): mm factor * 1000.
                pixel_size_um = _convert(voxel[0], _LENGTH_TO_MM[self.voxel_unit] * 1000, "voxel size")
        if pixel_size_um is not None:
            # rx_soln: the XML voxel size IS the pixel size, so set both together.
            # (derivers.py only maps voxel_size -> detector_pixel_size, not the reverse.)
            bundle.set("voxel_size", pixel_size_um, tier="mandatory_soft", source="file")
            bundle.set("detector_pixel_size", pixel_size_um, tier="mandatory_soft", source="file")
        else:
            for key in ("voxel_size", "detector_pixel_size"):
                bundle.declare_missing(key, tier="mandatory_soft")
        size = _xyz(_one(volume, "size"), integer=True)
        if size is not None and min(size) < 0:
            raise ValueError("RX Solution volume dimensions must be non-negative")
        optional("volume_size", size)
        optional("volume_offset", _xyz(_one(volume, "offset")))
        optional("rx_sol_voxel_unit", self.voxel_unit, source="user")
        bundle.set("rx_sol_distance_unit", self.distance_unit, tier="optional", source="user")

        acquisition = _one(section, "acquisitionInfo", "acquisitioninfo")
        geometry = _one(acquisition, "geometry")
        for key, attribute in (("source_to_detector_distance", "sdd"), ("source_to_object_distance", "sod")):
            if geometry_type == "cone":
                bundle.declare_missing(key, tier="mandatory_soft")
            value = _number(geometry, attribute)
            if value is not None:
                if value < 0:
                    raise ValueError(f"RX Solution {attribute} must be non-negative")
                optional(f"rx_sol_{key}", value)
                optional(f"rx_sol_{attribute}", value)
                if self.distance_unit is not None and value > 0:
                    tier = "mandatory_soft" if geometry_type == "cone" else "optional"
                    bundle.set(key, _convert(value, _LENGTH_TO_MM[self.distance_unit], attribute), tier=tier)
                elif self.distance_unit is None:
                    notes.append(f"geometry@{attribute} retained in raw units; confirm DISTANCE_UNIT to populate {key} in mm.")
        if geometry is not None and "ssd" in geometry.attrib:
            # SSD is preserved literally; it is not silently treated as SDD.
            bundle.set("rx_sol_ssd", geometry.get("ssd"), tier="optional")
            notes.append("geometry@ssd retained as written; its meaning is not assumed to be source-to-detector distance (sdd).")
        xray = _one(acquisition, "xray")
        for key, attribute in (("source_voltage", "voltage"), ("source_current", "current")):
            optional(key, _number(xray, attribute))

        reconstruction = _one(section, "reconstruct")
        if reconstruction is not None:
            for key, attribute in (("reconstruction_method", "methode"), ("reconstruction_filter", "filter"), ("output_format", "format")):
                optional(key, reconstruction.get(attribute))
        optional("metadata_version", root.get("version"))
        optional("metadata_revision", root.get("xactrevision"))
        bundle.set("rx_sol_metadata_notes", notes, tier="optional", source="computed")
        return bundle

