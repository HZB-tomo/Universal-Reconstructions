"""
Fallback plugins with low-specificity signatures: they win detection
only when no real vendor plugin matches a file, since DataImporter
picks the highest-specificity match.
"""
from __future__ import annotations

from data_import.plugin import ImportPlugin, register_plugin
from data_import.path_map import PathMap
from data_import.signature import Hdf5Signature, TiffSignature
from core.metadata import MetadataBundle


def _has_any_3d_dataset(h5file) -> tuple[bool, str]:
    """Custom signature condition for GenericHdf5Plugin: does the file
    contain at least one 3-dimensional dataset."""
    found = []

    def visitor(name, obj):
        if hasattr(obj, "ndim") and obj.ndim == 3:
            found.append(name)

    h5file.visititems(visitor)
    return (len(found) > 0, "contains at least one dataset with ndim==3")


@register_plugin
class GenericHdf5Plugin(ImportPlugin):
    """
    Fallback for an HDF5 file that doesn't match a known vendor: if it
    contains a 3D dataset, treats it as the projections and marks
    voxel size and geometry as needing user input.
    """

    name = "generic_hdf5"
    format = "hdf5"
    signature = Hdf5Signature(custom_predicate=_has_any_3d_dataset)

    def get_path_map(self, handle) -> PathMap:
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
            bundle.declare_missing("projections_path", tier="mandatory_hard")

        bundle.declare_missing("voxel_size", tier="mandatory_soft")
        bundle.declare_missing("geometry_type", tier="mandatory_soft")
        return bundle


@register_plugin
class GenericTiffPlugin(ImportPlugin):
    """
    Fallback for a folder of TIFF files that doesn't match a known
    vendor convention: treats every .tif/.tiff file as a projection
    image, in sorted filename order.
    """

    name = "generic_tiff"
    format = "tiff"
    signature = TiffSignature(min_file_count=1)

    def get_path_map(self, handle) -> PathMap:
        return PathMap(projections="**/*.tif*")

    def parse_metadata(self, handle, path_map: PathMap) -> MetadataBundle:
        bundle = MetadataBundle()
        bundle.set("projections_path", path_map.projections, tier="mandatory_hard")
        bundle.declare_missing("voxel_size", tier="mandatory_soft")
        bundle.declare_missing("geometry_type", tier="mandatory_soft")
        bundle.declare_missing("angles", tier="mandatory_soft")
        return bundle
