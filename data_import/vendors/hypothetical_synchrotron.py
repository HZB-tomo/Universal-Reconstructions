"""
Vendor plugin for a hypothetical synchrotron beamline's HDF5 layout.

    /entry/instrument/detector/data          projections, 3D
    /entry/instrument/detector/flat_fields   flat fields, 3D
    /entry/instrument/sample/rotation_angle  angles, 1D
    /entry/instrument/monochromator/energy   energy, scalar
    /entry/instrument/detector/pixel_size    pixel size, scalar
    root attribute: beamline == "hypothetical_synchrotron"

Matching test-data generator:
tests/fixtures/generate_hypothetical_synchrotron.py.
"""
from __future__ import annotations

from data_import.plugin import ImportPlugin, register_plugin
from data_import.path_map import PathMap
from data_import.signature import Hdf5Signature
from core.metadata import MetadataBundle

_PROJECTIONS = "/entry/instrument/detector/data"
_FLAT_FIELDS = "/entry/instrument/detector/flat_fields"
_ANGLES = "/entry/instrument/sample/rotation_angle"
_ENERGY = "/entry/instrument/monochromator/energy"
_PIXEL_SIZE = "/entry/instrument/detector/pixel_size"


@register_plugin
class HypotheticalSynchrotronHdf5Plugin(ImportPlugin):
    name = "hypothetical_synchrotron_hdf5"
    format = "hdf5"

    signature = Hdf5Signature(
        required_paths=[_PROJECTIONS, _FLAT_FIELDS, _ANGLES, _ENERGY, _PIXEL_SIZE],
        required_attrs={"/": {"beamline": "hypothetical_synchrotron"}},
    )

    def get_path_map(self, handle) -> PathMap:
        return PathMap(
            projections=_PROJECTIONS,
            flats=_FLAT_FIELDS,
            angles=_ANGLES,
        )

    def parse_metadata(self, handle, path_map: PathMap) -> MetadataBundle:
        bundle = MetadataBundle()

        bundle.set("projections_path", path_map.projections, tier="mandatory_hard")
        bundle.set("flats_path", path_map.flats, tier="mandatory_hard")
        bundle.set("angles", handle[_ANGLES][()], tier="mandatory_soft")

        # This beamline always writes pixel size, so it is read from
        # the file rather than declared missing.
        bundle.set("voxel_size", handle[_PIXEL_SIZE][()], tier="mandatory_soft", source="file")

        bundle.declare_missing("geometry_type", tier="mandatory_soft")
        bundle.set("energy", handle[_ENERGY][()], tier="optional")
        bundle.declare_missing("center_of_rotation", tier="computed")

        return bundle
