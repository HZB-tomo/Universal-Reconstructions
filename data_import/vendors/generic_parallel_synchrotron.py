"""
Vendor plugin for a generic parallel-beam synchrotron HDF5 layout.
Paired with tests/fixtures/realistic_phantom/, which writes files in
this layout, so a generated phantom file can be imported through the
normal DataImporter pipeline.

Always has both flat and dark fields, and is always parallel-beam,
both known unconditionally from the vendor's convention.

    /entry/instrument/detector/data           projections, 3D
    /entry/instrument/detector/flat_fields    flat fields, 3D
    /entry/instrument/detector/dark_fields    dark fields, 3D
    /entry/instrument/sample/rotation_angle   angles, 1D, degrees
    /entry/instrument/detector/pixel_size     voxel size, scalar
    root attribute: beamline == "generic_parallel_synchrotron"
"""
from __future__ import annotations

from data_import.plugin import ImportPlugin, register_plugin
from data_import.path_map import PathMap
from data_import.signature import Hdf5Signature
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
        required_paths=[_PROJECTIONS, _FLAT_FIELDS, _DARK_FIELDS, _ANGLES, _PIXEL_SIZE],
        required_attrs={"/": {"beamline": "generic_parallel_synchrotron"}},
    )

    def get_path_map(self, handle) -> PathMap:
        return PathMap(
            projections=_PROJECTIONS,
            flats=_FLAT_FIELDS,
            darks=_DARK_FIELDS,
            angles=_ANGLES,
        )

    def parse_metadata(self, handle, path_map: PathMap) -> MetadataBundle:
        bundle = MetadataBundle()

        bundle.set("projections_path", path_map.projections, tier="mandatory_hard")
        bundle.set("flats_path", path_map.flats, tier="mandatory_hard")
        bundle.set("darks_path", path_map.darks, tier="mandatory_hard")

        bundle.set("angles", handle[_ANGLES][()], tier="mandatory_soft")
        bundle.set("voxel_size", handle[_PIXEL_SIZE][()], tier="mandatory_soft", source="file")
        bundle.set("geometry_type", "parallel", tier="mandatory_soft", source="file")

        bundle.declare_missing("center_of_rotation", tier="computed")
        bundle.declare_missing("tilt_angle", tier="computed")

        return bundle
