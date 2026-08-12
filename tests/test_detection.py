"""
Tests for the detection logic in DataImporter.detect(): confirms a
correct match wins, a fallback is used when nothing specific matches,
and both failure modes (no match at all / ambiguous tie) raise the
right exception with useful detail.
"""
import pytest

from data_import.importer import DataImporter
from data_import.errors import NoVendorMatchError, AmbiguousVendorMatchError
from data_import.plugin import ImportPlugin, register_plugin
from data_import.path_map import PathMap
from data_import.signature import Hdf5Signature
from core.metadata import MetadataBundle


def test_detects_synthetic_hdf5_plugin(synthetic_data):
    importer = DataImporter()
    plugin = importer.detect(synthetic_data["hdf5_path"], format="hdf5")
    assert plugin.name == "synthetic_hdf5"


def test_detects_synthetic_tiff_plugin(synthetic_data):
    importer = DataImporter()
    plugin = importer.detect(synthetic_data["tiff_path"], format="tiff")
    assert plugin.name == "synthetic_tiff"


def test_falls_back_to_generic_hdf5_when_no_specific_plugin_matches(tmp_path):
    import h5py
    import numpy as np

    path = tmp_path / "unknown_vendor.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("some/random/path", data=np.zeros((5, 5, 5)))

    importer = DataImporter()
    plugin = importer.detect(path, format="hdf5")
    assert plugin.name == "generic_hdf5"


def test_no_match_raises_with_details(tmp_path):
    import h5py

    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("scalar", data=1)  # not 3D, so generic_hdf5 also fails

    importer = DataImporter()
    with pytest.raises(NoVendorMatchError) as exc_info:
        importer.detect(path, format="hdf5")
    assert exc_info.value.details  # per-plugin failure detail is present


def test_ambiguous_match_raises(tmp_path):
    import h5py
    import numpy as np

    path = tmp_path / "ambiguous.h5"
    with h5py.File(path, "w") as f:
        f.attrs["operator"] = "synthetic data"
        f.create_dataset("projections", data=np.zeros((2, 2, 2)))
        f.create_dataset("angles", data=np.zeros(2))

    # Register a second plugin with an identical signature/specificity
    # to synthetic_hdf5, to deliberately force a tie.
    @register_plugin
    class DuplicateSignaturePlugin(ImportPlugin):
        name = "duplicate_signature"
        format = "hdf5"
        signature = Hdf5Signature(
            required_paths=["/projections", "/angles"],
            required_attrs={"/": {"operator": "synthetic data"}},
        )

        def get_path_map(self, handle) -> PathMap:
            return PathMap(projections="/projections", angles="/angles")

        def parse_metadata(self, handle, path_map) -> MetadataBundle:
            return MetadataBundle()

    importer = DataImporter()
    with pytest.raises(AmbiguousVendorMatchError) as exc_info:
        importer.detect(path, format="hdf5")
    assert set(exc_info.value.candidates) == {"synthetic_hdf5", "duplicate_signature"}
