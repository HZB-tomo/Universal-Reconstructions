"""ILL neutron TIFF convention with scanInformation XML metadata.

The scan-root FlatsNeutrons and DarksNeutrons TIFFs identify the plugin.
Image loading belongs to the TIFF backend; projections use one file per frame.
Geometry is always parallel. Lengths are converted from XML millimetres to
registry micrometres for pixels and voxels. Angles are never inferred.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from core.metadata import MetadataBundle
from data_import.path_map import PathMap
from data_import.plugin import ImportPlugin, register_plugin
from data_import.signature import SignatureResult, TiffSignature

_PROJECTIONS = "Proj/proj_*.tif*"
_FLATS = "FlatsNeutrons.tif*"
_DARKS = "DarksNeutrons.tif*"
_METADATA = "scan_information.xml"
_REFERENCE_NAMES = (
    ("FlatsNeutrons.tif", "FlatsNeutrons.tiff"),
    ("DarksNeutrons.tif", "DarksNeutrons.tiff"),
)
_NUMBER = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
_VOXEL_PATTERN = re.compile(
    r"\s*" + r"\s*".join(fr"{axis}\s*:\s*{_NUMBER}\s*mm" for axis in "XYZ") + r"\s*"
)


def _number(node, attribute, *, integer=False):
    """Return None for absent values; reject malformed or non-finite numbers."""
    raw = node.get(attribute) if node is not None else None
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw) if integer else float(raw)
    except ValueError as error:
        raise ValueError(f"ILL invalid {node.tag}@{attribute}: {raw!r}") from error
    if not integer and not math.isfinite(value):
        raise ValueError(f"ILL non-finite {node.tag}@{attribute}: {raw!r}")
    return value


def _voxel_xyz_mm(text):
    """Parse an XYZ millimetre triple; blank values are unspecified."""
    if text is None or not text.strip():
        return None
    match = _VOXEL_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(f"ILL invalid reconstruction@voxelSize_mm: {text!r}")
    values = tuple(float(value) for value in match.groups())
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError(f"ILL invalid reconstruction@voxelSize_mm: {text!r}")
    return values


def _tiff_count(folder, pattern):
    """Reject non-TIFF matches that the backend's broad glob could load."""
    if not pattern:
        return 0
    count = 0
    for path in folder.glob(pattern):
        if not path.is_file():
            continue
        if path.suffix not in (".tif", ".tiff"):
            raise ValueError(f"ILL non-TIFF file matches {pattern!r}: {path.name}")
        count += 1
    return count


class ILLTiffSignature(TiffSignature):
    """Match exact, case-sensitive reference filenames at the scan root."""

    def check(self, folder):
        folder = Path(folder)
        checked, failed = [], []
        try:
            names = {path.name for path in folder.iterdir() if path.is_file()}
        except OSError as error:
            return SignatureResult(False, ["scan folder is readable"], [str(error)])
        for alternatives in _REFERENCE_NAMES:
            description = "scan-root file: " + " or ".join(alternatives)
            checked.append(description)
            if not names.intersection(alternatives):
                failed.append("missing " + description)
        return SignatureResult(not failed, checked, failed)


@register_plugin
class ILLTiffPlugin(ImportPlugin):
    name = "ill_neutrons"
    format = "tiff"
    signature = ILLTiffSignature(
        required_sidecar_patterns=[_FLATS, _DARKS],
        min_file_count=1,
    )

    def get_path_map(self, handle) -> PathMap:
        return PathMap(
            projections=_PROJECTIONS,
            flats=_FLATS,
            darks=_DARKS,
            metadata_source=_METADATA,
        )

    def parse_metadata(self, handle, path_map: PathMap) -> MetadataBundle:
        folder = Path(handle)
        if not path_map.metadata_source:
            raise ValueError("ILL metadata_source must name the scanInformation XML file.")
        metadata_path = folder / path_map.metadata_source
        try:
            root = ET.parse(metadata_path).getroot()
        except (OSError, ET.ParseError) as error:
            raise ValueError(f"ILL cannot read metadata {metadata_path}: {error}") from error
        if root.tag != "scanInformation":
            raise ValueError("ILL metadata root must be <scanInformation>.")
        for tag in ("imager", "acquisition", "geometry", "reconstruction", "manufacturer"):
            if len(root.findall(tag)) > 1:
                raise ValueError(f"ILL metadata has multiple <{tag}> sections.")

        bundle = MetadataBundle()
        counts = {}
        for key in ("projections", "flats", "darks"):
            pattern = getattr(path_map, key)
            counts[key] = _tiff_count(folder, pattern)
            if counts[key]:
                bundle.set(f"{key}_path", pattern, tier="mandatory_hard")
            else:
                bundle.declare_missing(f"{key}_path", tier="mandatory_hard")
        bundle.set("projection_count", counts["projections"], tier="optional", source="computed")

        def set_optional(key, value):
            if value is not None and (not isinstance(value, str) or value.strip()):
                bundle.set(key, value, tier="optional", source="file")

        # Retain vendor attributes, including placeholders and unconfirmed units.
        set_optional("ill_xml_attributes", [
            {"element": node.tag, "attributes": dict(node.attrib)} for node in root.iter()
        ])
        set_optional("project_name", root.get("projectName"))
        bundle.set("geometry_type", "parallel", tier="mandatory_soft", source="computed")
        for key in ("angles", "angle_span", "voxel_size", "detector_pixel_size"):
            bundle.declare_missing(key, tier="mandatory_soft")
        for key in ("center_of_rotation", "tilt_angle"):
            bundle.declare_missing(key, tier="computed")
        for key in ("energy", "acquisition_datetime"):
            bundle.declare_missing(key, tier="optional")

        acquisition = root.find("acquisition")
        declared = _number(acquisition, "numberOfProjections", integer=True)
        if declared is not None:
            if declared <= 0 or declared != counts["projections"]:
                raise ValueError(
                    f"ILL projection-count mismatch: XML declares {declared}, "
                    f"but {counts['projections']} files were found."
                )
            set_optional("ill_declared_images", declared)
        if acquisition is not None:
            set_optional("acquisition_method", acquisition.get("method"))
            duration = _number(acquisition, "duration_s")
            if duration is not None and duration < 0:
                raise ValueError("ILL acquisition@duration_s must be non-negative.")
            set_optional("acquisition_duration_s", duration)
            # Numeric date_iso values have no confirmed epoch or time zone.
            set_optional("ill_acquisition_date_iso", acquisition.get("date_iso"))

        imager = root.find("imager")
        pixel_um = None
        pixel_mm = _number(imager, "pixelSize_mm")
        if pixel_mm is not None:
            if pixel_mm < 0:
                raise ValueError("ILL imager@pixelSize_mm must be non-negative.")
            if pixel_mm > 0:
                pixel_um = pixel_mm * 1000.0
                if not math.isfinite(pixel_um):
                    raise ValueError("ILL pixel size overflows micrometre conversion.")
                bundle.set("detector_pixel_size", pixel_um, tier="mandatory_soft", source="file")
        width = _number(imager, "width", integer=True)
        height = _number(imager, "height", integer=True)
        if any(value is not None and value < 0 for value in (width, height)):
            raise ValueError("ILL imager width and height must be non-negative.")
        if width and height:
            set_optional("ill_declared_image_shape", (height, width))
        if imager is not None:
            # integrationTime has no declared unit.
            set_optional("ill_integration_time", imager.get("integrationTime"))

        reconstruction = root.find("reconstruction")
        voxel = None
        if reconstruction is not None:
            voxel = _voxel_xyz_mm(reconstruction.get("voxelSize_mm"))
            if voxel is not None:
                set_optional("ill_voxel_size_xyz_mm", voxel)
                # Zero is unset; anisotropic triples cannot populate a scalar field.
                if min(voxel) > 0 and all(
                    math.isclose(value, voxel[0], rel_tol=1e-6, abs_tol=0.0) for value in voxel
                ):
                    voxel_um = voxel[0] * 1000.0
                    if not math.isfinite(voxel_um):
                        raise ValueError("ILL voxel size overflows micrometre conversion.")
                    bundle.set("voxel_size", voxel_um, tier="mandatory_soft", source="file")
            set_optional("reconstruction_method", reconstruction.get("method", "").strip())
        # Parallel-beam sampling uses the calibrated image pixel size when
        # reconstruction spacing is unspecified. Preserve explicit XYZ values.
        if pixel_um is not None and (voxel is None or all(value == 0 for value in voxel)):
            bundle.set("voxel_size", pixel_um, tier="mandatory_soft", source="file")
        manufacturer = root.find("manufacturer")
        if manufacturer is not None:
            set_optional("instrument_name", manufacturer.get("deviceName", "").strip())
        return bundle
