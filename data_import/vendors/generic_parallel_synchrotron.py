"""
Vendor plugins for a generic parallel-beam synchrotron layout, HDF5 and
TIFF. Paired with tests/fixtures/realistic_phantom/, which writes files
in these layouts (write_hdf5()/write_tiff()), so a generated phantom
can be imported through the normal DataImporter pipeline.

Always parallel-beam, known unconditionally from the vendor's convention.

HDF5 (only the projections and the root attribute are required):
    /entry/instrument/detector/data           projections, 3D
    /entry/instrument/detector/flat_fields    flat fields, 3D (optional)
    /entry/instrument/detector/dark_fields    dark fields, 3D (optional)
    /entry/instrument/sample/rotation_angle   angles, 1D, degrees (optional)
    /entry/instrument/detector/pixel_size     voxel size, scalar (optional)
    root attribute: beamline == "generic_parallel_synchrotron"

TIFF (a folder):
    projections/*.tif   projections, one file per angle
    flats/*.tif         flat fields, one file per frame
    darks/*.tif         dark fields, one file per frame
    metadata.json        {"angles": [...], "voxel_size": ...}
"""
from __future__ import annotations

import json

import numpy as np

from data_import.plugin import ImportPlugin, register_plugin
from data_import.path_map import PathMap
from data_import.signature import Hdf5Signature, TiffSignature
from core.metadata import MetadataBundle

_PROJECTIONS = "/entry/instrument/detector/data"
_FLAT_FIELDS = "/entry/instrument/detector/flat_fields"
_DARK_FIELDS = "/entry/instrument/detector/dark_fields"
_ANGLES = "/entry/instrument/sample/rotation_angle"
_PIXEL_SIZE = "/entry/instrument/detector/pixel_size"


@register_plugin
class GenericParallelSynchrotronHdf5Plugin(ImportPlugin):
    name = "generic_parallel_synchrotron_hdf5"
    format = "hdf5"

    signature = Hdf5Signature(
        required_paths=[_PROJECTIONS],
        required_attrs={"/": {"beamline": "generic_parallel_synchrotron"}},
    )

    def get_path_map(self, handle) -> PathMap:
        return PathMap(
            projections=_PROJECTIONS,
            flats=_FLAT_FIELDS if _FLAT_FIELDS in handle else None,
            darks=_DARK_FIELDS if _DARK_FIELDS in handle else None,
            angles=_ANGLES if _ANGLES in handle else None,
        )

    def parse_metadata(self, handle, path_map: PathMap) -> MetadataBundle:
        bundle = MetadataBundle()

        bundle.set("projections_path", path_map.projections, tier="mandatory_hard")
        if path_map.flats:
            bundle.set("flats_path", path_map.flats, tier="mandatory_hard")
        if path_map.darks:
            bundle.set("darks_path", path_map.darks, tier="mandatory_hard")

        if _ANGLES in handle:
            bundle.set("angles", handle[_ANGLES][()], tier="mandatory_soft")
        else:
            # Optional: a converter may not know the angles. Left to the
            # completion stage (derive from angle_span, or ask).
            bundle.declare_missing("angles", tier="mandatory_soft")
        if _PIXEL_SIZE in handle:
            bundle.set("voxel_size", handle[_PIXEL_SIZE][()], tier="mandatory_soft", source="file")
        else:
            bundle.declare_missing("voxel_size", tier="mandatory_soft")
        bundle.set("geometry_type", "parallel", tier="mandatory_soft", source="file")

        bundle.declare_missing("center_of_rotation", tier="computed")
        bundle.declare_missing("tilt_angle", tier="computed")

        return bundle


@register_plugin
class GenericParallelSynchrotronTiffPlugin(ImportPlugin):
    name = "generic_parallel_synchrotron_tiff"
    format = "tiff"

    signature = TiffSignature(
        required_sidecar_patterns=["metadata.json", "flats/*.tif*", "darks/*.tif*"],
        required_sidecar_keys={"metadata.json": ["angles", "voxel_size"]},
        min_file_count=1,
    )

    def get_path_map(self, handle) -> PathMap:
        return PathMap(
            projections="projections/*.tif*",
            flats="flats/*.tif*",
            darks="darks/*.tif*",
            metadata_source="metadata.json",
        )

    def parse_metadata(self, handle, path_map: PathMap) -> MetadataBundle:
        bundle = MetadataBundle()

        bundle.set("projections_path", path_map.projections, tier="mandatory_hard")
        bundle.set("flats_path", path_map.flats, tier="mandatory_hard")
        bundle.set("darks_path", path_map.darks, tier="mandatory_hard")

        # `handle` is the folder Path (see backend.py's TiffBackend.open).
        sidecar = next(handle.rglob(path_map.metadata_source), None)
        data = json.loads(sidecar.read_text()) if sidecar is not None else {}

        angles = data.get("angles")
        bundle.set("angles", np.asarray(angles) if angles is not None else None, tier="mandatory_soft", source="file")
        bundle.set("voxel_size", data.get("voxel_size"), tier="mandatory_soft", source="file")
        bundle.set("geometry_type", "parallel", tier="mandatory_soft", source="file")

        bundle.declare_missing("center_of_rotation", tier="computed")
        bundle.declare_missing("tilt_angle", tier="computed")

        return bundle
