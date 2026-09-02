"""
These two plugins are our reference implementation -- the one concrete,
real example of "a plugin for a specific data convention" that this
codebase has right now. They match the synthetic test fixture generated
by tests/fixtures/generate_synthetic.py:

  HDF5 convention:
      /projections           <- dataset, shape (n_angles, height, width)
      /angles                <- dataset, shape (n_angles,)
      root attribute "operator" == "synthetic data"

  TIFF convention:
      synthetic_data/
          projections/*.tif  <- one file per angle
          metadata.json      <- {"angles": [...], "operator": "synthetic data"}

Use these as the template when you write a plugin for a REAL instrument
later: copy one of these two classes, change the signature to match your
vendor's actual paths/attributes/sidecar format, and change
get_path_map()/parse_metadata() accordingly. Everything else (detection,
specificity, how DataImporter uses your plugin) works automatically --
you never need to touch importer.py, backend.py, or core/.
"""
from __future__ import annotations

import json

from data_import.plugin import ImportPlugin, register_plugin
from data_import.path_map import PathMap
from data_import.signature import Hdf5Signature, TiffSignature
from core.metadata import MetadataBundle


@register_plugin
class SyntheticHdf5Plugin(ImportPlugin):
    name = "synthetic_hdf5"
    format = "hdf5"

    # 3 required paths would be enough, but we also require the root
    # attribute "operator" == "synthetic data" so this signature is
    # unambiguously specific to OUR synthetic fixtures and won't
    # accidentally match some unrelated real vendor's hdf5 file that
    # happens to also have "/projections" and "/angles" datasets.
    signature = Hdf5Signature(
        required_paths=["/projections", "/angles"],
        required_attrs={"/": {"operator": "synthetic data"}},
    )

    def get_path_map(self, handle) -> PathMap:
        # Straightforward, because our synthetic convention is simple:
        # the paths are fixed, no searching needed.
        return PathMap(projections="/projections", angles="/angles")

    def parse_metadata(self, handle, path_map: PathMap) -> MetadataBundle:
        bundle = MetadataBundle()

        # mandatory_hard: without projections there's nothing to work
        # with at all. angles is mandatory_soft -- if a vendor omits it,
        # completion can still derive or ask for it later, so it
        # shouldn't block import outright.
        bundle.set("projections_path", path_map.projections, tier="mandatory_hard")
        bundle.set("angles", handle["/angles"][()], tier="mandatory_soft")

        # optional: nice to record, nothing depends on it.
        bundle.set("operator", handle.attrs.get("operator"), tier="optional")

        # mandatory_soft: our synthetic fixtures don't encode voxel size
        # or geometry type at all, so these are always missing and
        # always need to be prompted for -- a real vendor plugin would
        # instead try to read these from the file first, and only mark
        # them missing if they're genuinely absent (see GenericHdf5Plugin
        # for an example of always-missing, and imagine a real vendor
        # plugin conditionally reading them, as sketched in our earlier
        # discussion of a hypothetical Zeiss plugin).
        bundle.declare_missing("voxel_size", tier="mandatory_soft")
        bundle.declare_missing("geometry_type", tier="mandatory_soft")

        # computed: no starting guess available from our synthetic
        # fixtures, so declared missing here; a real reconstruction
        # module would fill this in later via DataInterface.set_computed().
        bundle.declare_missing("center_of_rotation", tier="computed")

        return bundle


@register_plugin
class SyntheticTiffPlugin(ImportPlugin):
    name = "synthetic_tiff"
    format = "tiff"

    signature = TiffSignature(
        required_sidecar_patterns=["metadata.json"],
        required_sidecar_keys={"metadata.json": ["angles", "operator"]},
        min_file_count=1,
    )

    def get_path_map(self, handle) -> PathMap:
        return PathMap(
            projections="projections/*.tif*",
            metadata_source="metadata.json",
        )

    def parse_metadata(self, handle, path_map: PathMap) -> MetadataBundle:
        bundle = MetadataBundle()
        bundle.set("projections_path", path_map.projections, tier="mandatory_hard")

        # `handle` here is the folder Path (see backend.py's
        # TiffBackend.open) -- find the sidecar file the signature check
        # already confirmed exists, and read it.
        sidecar = next(handle.rglob(path_map.metadata_source), None)
        if sidecar is None:
            # Defensive fallback -- shouldn't happen since the signature
            # already required this sidecar to exist.
            bundle.declare_missing("angles", tier="mandatory_soft")
            bundle.declare_missing("operator", tier="optional")
        else:
            data = json.loads(sidecar.read_text())
            bundle.set("angles", data.get("angles"), tier="mandatory_soft")
            bundle.set("operator", data.get("operator"), tier="optional")

        bundle.declare_missing("voxel_size", tier="mandatory_soft")
        bundle.declare_missing("geometry_type", tier="mandatory_soft")
        bundle.declare_missing("center_of_rotation", tier="computed")
        return bundle
