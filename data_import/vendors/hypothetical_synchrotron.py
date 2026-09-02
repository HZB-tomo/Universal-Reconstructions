"""
Vendor plugin for a hypothetical synchrotron beamline's HDF5 layout.
This is our second real (non-fallback, non-synthetic-test) plugin, and
it exists mainly as a second worked example alongside synthetic.py --
notice it needed ZERO changes to signature.py, plugin.py, backend.py, or
importer.py. That's the whole point of the plugin contract: a new
instrument is just a new file in vendors/, plus one import line in
vendors/__init__.py.

Assumed file layout (see project discussion for how this was decided):

    /entry/instrument/detector/data          projections, 3D
    /entry/instrument/detector/flat_fields   flat fields, 3D
    /entry/instrument/sample/rotation_angle  angles, 1D
    /entry/instrument/monochromator/energy   energy, scalar
    /entry/instrument/detector/pixel_size    pixel size, scalar
    root attribute: beamline == "hypothetical_synchrotron"

A matching test-data generator lives at
tests/fixtures/generate_hypothetical_synchrotron.py.
"""
from __future__ import annotations

from data_import.plugin import ImportPlugin, register_plugin
from data_import.path_map import PathMap
from data_import.signature import Hdf5Signature
from core.metadata import MetadataBundle

# Internal hdf5 paths, named once here so get_path_map() and
# parse_metadata() below both reference the same constants instead of
# repeating the literal strings (and risking a typo mismatch between them).
_PROJECTIONS = "/entry/instrument/detector/data"
_FLAT_FIELDS = "/entry/instrument/detector/flat_fields"
_ANGLES = "/entry/instrument/sample/rotation_angle"
_ENERGY = "/entry/instrument/monochromator/energy"
_PIXEL_SIZE = "/entry/instrument/detector/pixel_size"


@register_plugin
class HypotheticalSynchrotronHdf5Plugin(ImportPlugin):
    name = "hypothetical_synchrotron_hdf5"
    format = "hdf5"

    # 5 required paths + 1 required attribute = specificity 6. This is
    # higher than synthetic_hdf5's specificity of 3 and generic_hdf5's
    # specificity of 1, so on a real file from THIS beamline, this
    # plugin will always be picked over those -- and since the required
    # paths are specific to this layout, it won't accidentally match
    # files from other vendors either.
    signature = Hdf5Signature(
        required_paths=[_PROJECTIONS, _FLAT_FIELDS, _ANGLES, _ENERGY, _PIXEL_SIZE],
        required_attrs={"/": {"beamline": "hypothetical_synchrotron"}},
    )

    def get_path_map(self, handle) -> PathMap:
        # Just point at the fixed paths -- no searching needed since this
        # vendor's layout is fully known and constant.
        return PathMap(
            projections=_PROJECTIONS,
            flats=_FLAT_FIELDS,
            angles=_ANGLES,
        )

    def parse_metadata(self, handle, path_map: PathMap) -> MetadataBundle:
        bundle = MetadataBundle()

        # mandatory_hard: without projections/flats there's nothing to
        # reconstruct from. angles is mandatory_soft instead -- even
        # though this beamline always provides it, angles in general is
        # recoverable later (derived from span + projection count, or
        # asked for) rather than a hard import blocker.
        bundle.set("projections_path", path_map.projections, tier="mandatory_hard")
        bundle.set("flats_path", path_map.flats, tier="mandatory_hard")
        bundle.set("angles", handle[_ANGLES][()], tier="mandatory_soft")

        # mandatory_soft: voxel_size normally needs prompting for other
        # vendors, but THIS beamline always writes it -- so we read it
        # straight from the file. The field still belongs to the
        # mandatory_soft tier (that's what "voxel_size" means across the
        # whole software), it's just never actually missing for this
        # particular vendor.
        bundle.set("voxel_size", handle[_PIXEL_SIZE][()], tier="mandatory_soft", source="file")

        # This vendor still never tells us the geometry type (parallel
        # vs cone-beam, say) -- so exactly like the synthetic and
        # generic plugins, we declare it missing and let the user
        # supply it.
        bundle.declare_missing("geometry_type", tier="mandatory_soft")

        # optional: nice to have, nothing downstream depends on it yet.
        bundle.set("energy", handle[_ENERGY][()], tier="optional")

        # computed: no starting guess available from this beamline's
        # files, same as our other plugins.
        bundle.declare_missing("center_of_rotation", tier="computed")

        return bundle
