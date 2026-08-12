"""
These two plugins are deliberately "dumb" fallbacks: they use very
generic signatures (low specificity, i.e. very few required conditions),
so they only ever win detection when NO real vendor plugin matches a
file. This gives the software something to fall back on for unrecognized
data, rather than refusing to load anything it doesn't have specific
vendor knowledge about.

Concretely: any real vendor plugin (like SyntheticHdf5Plugin below in
synthetic.py, which checks 5 conditions) will always outrank these
(which check only 1 condition), because DataImporter picks the HIGHEST
specificity match. These only "win" when they're the only match at all.
"""
from __future__ import annotations

from data_import.plugin import ImportPlugin, register_plugin
from data_import.path_map import PathMap
from data_import.signature import Hdf5Signature, TiffSignature
from core.metadata import MetadataBundle


def _has_any_3d_dataset(h5file) -> tuple[bool, str]:
    """
    Custom signature condition for GenericHdf5Plugin: rather than
    checking a specific known path (which we don't have, since this is
    the fallback for UNKNOWN vendors), just check that the file contains
    at least one 3-dimensional dataset anywhere -- a reasonable proxy for
    "this probably contains a projection stack".
    """
    found = []

    def visitor(name, obj):
        if hasattr(obj, "ndim") and obj.ndim == 3:
            found.append(name)

    h5file.visititems(visitor)
    return (len(found) > 0, "contains at least one dataset with ndim==3")


@register_plugin
class GenericHdf5Plugin(ImportPlugin):
    """
    Fallback for any HDF5 file that doesn't match a known vendor: if it
    contains at least one 3D dataset, treat that as "probably the
    projections" and let the user sort out the rest (voxel size,
    geometry, etc. all get marked mandatory_soft, prompting the user).
    """

    name = "generic_hdf5"
    format = "hdf5"
    signature = Hdf5Signature(custom_predicate=_has_any_3d_dataset)

    def get_path_map(self, handle) -> PathMap:
        # Find the first 3D dataset again (same search as the signature
        # check) and assume it's the projections. Best-effort only --
        # this is the fallback path, not a confident vendor-specific read.
        found = []

        def visitor(name, obj):
            if hasattr(obj, "ndim") and obj.ndim == 3:
                found.append(name)

        handle.visititems(visitor)
        return PathMap(projections=found[0] if found else None)

    def parse_metadata(self, handle, path_map: PathMap) -> MetadataBundle:
        bundle = MetadataBundle()
        if path_map.projections:
            bundle.set("projections_path", path_map.projections, tier="mandatory_hard")
        else:
            # Shouldn't normally happen (the signature already required
            # a 3D dataset to exist), but handled defensively anyway.
            bundle.declare_missing("projections_path", tier="mandatory_hard")

        # We have no way to know voxel size or geometry for an unknown
        # vendor -- always ask the user for these.
        bundle.declare_missing("voxel_size", tier="mandatory_soft")
        bundle.declare_missing("geometry_type", tier="mandatory_soft")
        return bundle


@register_plugin
class GenericTiffPlugin(ImportPlugin):
    """
    Fallback for any folder of TIFF files that doesn't match a known
    vendor convention: just treat every .tif/.tiff file in the folder as
    a projection image, in sorted filename order.
    """

    name = "generic_tiff"
    format = "tiff"
    signature = TiffSignature(min_file_count=1)

    def get_path_map(self, handle) -> PathMap:
        return PathMap(projections="**/*.tif*")

    def parse_metadata(self, handle, path_map: PathMap) -> MetadataBundle:
        bundle = MetadataBundle()
        bundle.set("projections_path", path_map.projections, tier="mandatory_hard")
        # No sidecar metadata convention is known for an unrecognized
        # vendor -- everything else has to come from the user.
        bundle.declare_missing("voxel_size", tier="mandatory_soft")
        bundle.declare_missing("geometry_type", tier="mandatory_soft")
        bundle.declare_missing("angles", tier="mandatory_soft")
        return bundle
