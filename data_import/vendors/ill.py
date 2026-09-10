"""ILL TIFF/XML convention: paths, identity, and metadata only.

Like generic.py, this plugin leaves image loading to the TIFF backend.
The supplied XML has no angle list or explicit length units. Unknown values
remain soft-missing; XML values are retained under ill_* names for inspection.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from core.metadata import MetadataBundle
from data_import.path_map import PathMap
from data_import.plugin import ImportPlugin, register_plugin
from data_import.signature import SignatureResult, TiffSignature

_PROJECTIONS = "Proj/proj_*.tif*"
_FLATS = "Flats/flat_*.tif*"
_DARKS = "Darks/dark_*.tif*"
_METADATA = "unireconstruction.xml"
_ILL_TOKEN = re.compile(r"(?<![A-Za-z0-9])ILL(?![A-Za-z0-9])", re.IGNORECASE)
_LENGTH_TO_MM = {"m": 1000.0, "mm": 1.0, "um": 0.001}


def _xyz(node, value_type=float):
    if node is None:
        return None
    try:
        values = tuple(value_type(node.attrib[axis]) for axis in ("X", "Y", "Z"))
    except (KeyError, TypeError, ValueError):
        return None
    return values if np.all(np.isfinite(values)) else None


class ILLTiffSignature(TiffSignature):
    """Recognize the ILL XML convention, not merely any TIFF folder."""

    @property
    def specificity(self):
        return super().specificity + 2

    def check(self, folder):
        folder = Path(folder)
        result = super().check(folder)
        # Use the same scan root as PathMap; base sidecar checks are recursive.
        for pattern in self.required_sidecar_patterns:
            if not any(p.is_file() for p in folder.glob(pattern)):
                result.failed.append(f"missing file relative to scan root: {pattern}")
        result.checked.extend([
            "XML root is unireconstruction",
            "XML xactrevision contains the ILL token",
        ])
        try:
            root = ET.parse(folder / _METADATA).getroot()
        except (OSError, ET.ParseError) as error:
            result.failed.append(f"cannot read ILL metadata: {error}")
        else:
            if root.tag != "unireconstruction":
                result.failed.append("XML root is not unireconstruction")
            if not _ILL_TOKEN.search(root.get("xactrevision", "")):
                result.failed.append("XML xactrevision has no ILL token")
        return SignatureResult(not result.failed, result.checked, result.failed)


@register_plugin
class ILLTiffPlugin(ImportPlugin):
    name = "ill_tiff"
    format = "tiff"
    signature = ILLTiffSignature(
        required_sidecar_patterns=[_PROJECTIONS, _FLATS, _DARKS, _METADATA],
        min_file_count=1,
    )

    def __init__(self, *, voxel_unit=None, distance_unit=None):
        """Supply units only after confirming the XML producer's convention.

        Each accepts 'm', 'mm', 'um', or None (unknown). Voxel and source
        distances can have different units. Automatic detection uses None.
        """
        for unit in (voxel_unit, distance_unit):
            if unit is not None and unit not in _LENGTH_TO_MM:
                raise ValueError("ILL length units must be 'm', 'mm', 'um', or None.")
        self.voxel_unit = voxel_unit
        self.distance_unit = distance_unit

    def get_path_map(self, handle) -> PathMap:
        return PathMap(
            projections=_PROJECTIONS, flats=_FLATS, darks=_DARKS,
            metadata_source=_METADATA,
        )

    def parse_metadata(self, handle, path_map: PathMap) -> MetadataBundle:
        folder = Path(handle)
        bundle = MetadataBundle()
        for key in ("projections", "flats", "darks"):
            pattern = getattr(path_map, key)
            if pattern and any(p.is_file() for p in folder.glob(pattern)):
                bundle.set(f"{key}_path", pattern, tier="mandatory_hard")
            else:
                bundle.declare_missing(f"{key}_path", tier="mandatory_hard")

        root = ET.parse(folder / path_map.metadata_source).getroot()
        if root.tag != "unireconstruction":
            raise ValueError("ILL metadata root must be <unireconstruction>.")
        sections = [
            (name, node)
            for name, tag in (("cone", "conebeam"), ("parallel", "parallelbeam"))
            for node in root.findall(tag)
        ]
        if len(sections) != 1:
            raise ValueError(
                "ILL metadata must contain exactly one <conebeam> or <parallelbeam> section."
            )
        geometry_type, section = sections[0]
        bundle.set("geometry_type", geometry_type, tier="mandatory_soft")

        count = (sum(p.is_file() for p in folder.glob(path_map.projections))
                 if path_map.projections else 0)
        bundle.set("projection_count", count, tier="optional", source="computed")
        profile = section.find("profile")
        if profile is not None:
            if profile.get("images") is not None:
                declared = int(profile.get("images"))
                bundle.set("ill_declared_images", declared, tier="optional")
                if count and (declared <= 0 or declared != count):
                    raise ValueError(
                        f"ILL projection-count mismatch: XML declares {declared}, "
                        f"but {count} files were found."
                    )
            if profile.get("turns") is not None:
                bundle.set("turns", float(profile.get("turns")), tier="optional")
        # Neither turns nor the reconstruction profile establishes real angles.
        bundle.declare_missing("angles", tier="mandatory_soft")
        bundle.declare_missing("angle_span", tier="mandatory_soft")
        bundle.declare_missing("voxel_size", tier="mandatory_soft")
        bundle.declare_missing("detector_pixel_size", tier="mandatory_soft")

        volume = section.find("volume")
        if volume is not None:
            voxel = _xyz(volume.find("voxelSize"))
            if voxel is not None:
                bundle.set("ill_voxel_size_xyz", voxel, tier="optional")
                # The current reconstruction API expects one scalar in um.
                # Preserve anisotropic or unitless triples without guessing.
                if (self.voxel_unit is not None and min(voxel) > 0
                        and np.allclose(voxel, voxel[0], rtol=1e-6, atol=0)):
                    bundle.set("voxel_size", voxel[0] * _LENGTH_TO_MM[self.voxel_unit]
                               * 1000.0, tier="mandatory_soft")
            for key, tag, cast in (("volume_size", "size", int),
                                   ("volume_offset", "offset", float)):
                value = _xyz(volume.find(tag), cast)
                if value is not None:
                    bundle.set(key, value, tier="optional")
        bundle.set("ill_voxel_unit", self.voxel_unit, tier="optional", source="user")
        bundle.set("ill_distance_unit", self.distance_unit, tier="optional", source="user")

        acquisition = section.find("acquisitioninfo")
        for key, attribute in (("source_to_detector_distance", "sdd"),
                               ("source_to_object_distance", "sod")):
            if geometry_type == "cone":
                bundle.declare_missing(key, tier="mandatory_soft")
            node = acquisition.find("geometry") if acquisition is not None else None
            if node is not None and node.get(attribute) is not None:
                value = float(node.get(attribute))
                bundle.set(f"ill_{key}", value, tier="optional")
                if (geometry_type == "cone" and self.distance_unit is not None
                        and np.isfinite(value) and value > 0):
                    bundle.set(key, value * _LENGTH_TO_MM[self.distance_unit],
                               tier="mandatory_soft")
        if acquisition is not None:
            xray = acquisition.find("xray")
            for key, attribute in (("source_voltage", "voltage"),
                                   ("source_current", "current")):
                if xray is not None and xray.get(attribute) is not None:
                    bundle.set(key, float(xray.get(attribute)), tier="optional")

        reconstruction = section.find("reconstruct")
        if reconstruction is not None:
            for key, attribute in (("reconstruction_method", "methode"),
                                   ("reconstruction_filter", "filter"),
                                   ("output_format", "format")):
                bundle.set(key, reconstruction.get(attribute), tier="optional")
        bundle.set("metadata_version", root.get("version"), tier="optional")
        bundle.set("metadata_revision", root.get("xactrevision"), tier="optional")
        bundle.declare_missing("center_of_rotation", tier="computed")
        bundle.declare_missing("tilt_angle", tier="computed")
        return bundle
