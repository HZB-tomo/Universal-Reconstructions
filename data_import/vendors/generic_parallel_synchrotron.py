"""
Vendor plugin for a "generic parallel-beam synchrotron" hdf5 layout.
Paired with tests/fixtures/realistic_phantom/, which writes files in
exactly this layout -- so a generated phantom file can be imported
through the normal DataImporter pipeline (plugin detection, reconcile(),
the works) exactly like a real vendor file, not just consumed as an
in-memory shortcut.

Deliberately a separate plugin from hypothetical_synchrotron_hdf5
rather than an extension of it: that plugin never modeled a dark field
at all (its "Assumed file layout" simply doesn't include one), and
retrofitting dark-field support onto an existing, already-tested vendor
plugin just to serve a new test fixture seemed like the wrong place to
absorb that change. This vendor always has both flat AND dark fields,
and is always parallel-beam -- both known unconditionally from the
vendor's own convention, the same way hypothetical_synchrotron_hdf5
already reads voxel_size unconditionally from file.

Assumed file layout:

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

    # 5 required paths + 1 required attribute = specificity 6. The
    # attribute value ("generic_parallel_synchrotron") is unique to this
    # vendor, so this can never tie with hypothetical_synchrotron_hdf5
    # (also specificity 6, different attribute value) on a real file --
    # a file's root attribute can only equal one string.
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

        # mandatory_hard: this vendor always provides all three arrays --
        # without any of them there's nothing to work with.
        bundle.set("projections_path", path_map.projections, tier="mandatory_hard")
        bundle.set("flats_path", path_map.flats, tier="mandatory_hard")
        bundle.set("darks_path", path_map.darks, tier="mandatory_hard")

        bundle.set("angles", handle[_ANGLES][()], tier="mandatory_soft")
        bundle.set("voxel_size", handle[_PIXEL_SIZE][()], tier="mandatory_soft", source="file")

        # Known unconditionally from the vendor's own convention -- this
        # beamline is always parallel-beam, so there's nothing to ask
        # the user for here, the same way voxel_size isn't asked for.
        bundle.set("geometry_type", "parallel", tier="mandatory_soft", source="file")

        # computed: no starting guess available from this vendor's files --
        # calibration's job to fill these in later.
        bundle.declare_missing("center_of_rotation", tier="computed")
        bundle.declare_missing("tilt_angle", tier="computed")

        return bundle
