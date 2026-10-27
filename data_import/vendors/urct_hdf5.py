"""
Import plugin for the URCT HDF5 file standard (doc/FILE_STANDARD.md): a
projection file written by the converter or by any tool that follows the
standard. Recognized by its root attributes, not by a vendor name.

Reads every known metadata field from its standard path (core/urct_format.py),
also from the older angle location, and carries /entry/extra/* along. A field
absent from the file stays absent; the completion stage then fills or asks.
Older converted files (the interim layout) are read by
generic_parallel_synchrotron_hdf5, which is kept for them.
"""
from __future__ import annotations

import warnings

import numpy as np

from core import urct_format as fmt
from core.known_fields import KNOWN_FIELDS
from core.metadata import MetadataBundle
from data_import.path_map import PathMap
from data_import.plugin import ImportPlugin, register_plugin
from data_import.signature import Hdf5Signature


@register_plugin
class UrctHdf5Plugin(ImportPlugin):
    name = "urct_hdf5"
    format = "hdf5"

    signature = Hdf5Signature(
        required_paths=[fmt.PROJECTIONS],
        required_attrs={"/": {"urct_standard": fmt.STANDARD, "urct_file_type": fmt.FILE_PROJECTIONS}},
    )

    def get_path_map(self, handle) -> PathMap:
        _check_version(handle)
        angles = fmt.find_field(handle, "angles")
        return PathMap(
            projections=fmt.PROJECTIONS,
            flats=fmt.FLAT_FIELDS if fmt.FLAT_FIELDS in handle else None,
            darks=fmt.DARK_FIELDS if fmt.DARK_FIELDS in handle else None,
            angles=angles[2] if angles is not None else None,
        )

    def parse_metadata(self, handle, path_map: PathMap) -> MetadataBundle:
        bundle = MetadataBundle()
        bundle.set("projections_path", path_map.projections, tier="mandatory_hard")
        if path_map.flats:
            bundle.set("flats_path", path_map.flats, tier="mandatory_hard")
        if path_map.darks:
            bundle.set("darks_path", path_map.darks, tier="mandatory_hard")

        for key, definition in KNOWN_FIELDS.items():
            found = fmt.find_field(handle, key)
            if found is None:
                continue
            value, source, _path = found
            bundle.set(key, value, tier=definition.tier, source=source if source in _SOURCES else "file")

        # Parallel beam: detector pixel and voxel are the same size, and a file may give just one.
        if bundle.get("voxel_size") is None and bundle.get("detector_pixel_size") is not None \
                and bundle.get("geometry_type") == "parallel":
            bundle.set("voxel_size", bundle.get("detector_pixel_size"), tier="mandatory_soft", source="derived")

        if handle.attrs.get("urct_uuid"):       # lets a volume made from this file point back at it
            bundle.set("source_file_uuid", str(handle.attrs["urct_uuid"]), tier="optional", source="file")

        # Anything under /entry/extra rides along, uninterpreted.
        extra = handle.get(fmt.EXTRA)
        if extra is not None:
            for name, node in extra.items():
                if hasattr(node, "shape") and node.size <= 1024:
                    bundle.set(f"extra.{name}", fmt.read_value(node), tier="optional", source="file")
        return bundle


_SOURCES = ("file", "user", "computed", "default", "derived")


def _check_version(handle) -> None:
    text = handle.attrs.get("urct_version", "0.0")
    major, minor = fmt.parse_version(text)
    if major > fmt.FORMAT_VERSION[0]:
        raise ValueError(
            f"this file follows URCT file standard {text}; this software understands "
            f"{fmt.FORMAT_VERSION[0]}.x. Update the software."
        )
    if (major, minor) > fmt.FORMAT_VERSION:
        warnings.warn(f"file standard {text} is newer than this software's {fmt.FORMAT_VERSION[0]}.{fmt.FORMAT_VERSION[1]}; unknown fields are ignored")
