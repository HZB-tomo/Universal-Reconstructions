"""
ImportPlugin: the contract every vendor plugin implements, and the
registry plugins register with.

A plugin answers three questions about one vendor's file convention:
  - Is this my file? -> `signature` (see signature.py)
  - Where's everything? -> get_path_map()
  - What does the metadata mean? -> parse_metadata()

A plugin never opens a file itself (FormatBackend's job, see
backend.py) and never chooses RAM vs disk (the caller's choice, via
DataImporter, see importer.py).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from data_import.signature import Signature, SignatureResult
from data_import.path_map import PathMap
from core.metadata import MetadataBundle


class ImportPlugin(ABC):
    """
    Base class for a vendor plugin.

    Class attributes to set on a subclass:
        name:      a short unique string, e.g. "zeiss_xradia". Used for
                   logging and ProjectObject.vendor_plugin_name.
        format:    "hdf5" or "tiff".
        signature: an instance of Hdf5Signature or TiffSignature
                   matching `format`. See signature.py.
    """

    name: str
    format: Literal["hdf5", "tiff"]
    signature: Signature

    def matches(self, target) -> SignatureResult:
        """
        Check whether `target` (an open h5py.File for hdf5 plugins, or
        a folder Path for tiff plugins) satisfies this plugin's
        signature. Delegates to self.signature.check().
        """
        return self.signature.check(target)

    @abstractmethod
    def get_path_map(self, handle) -> PathMap:
        """
        Called after this plugin has been selected as the best match
        (see importer.py). Returns a PathMap pointing to where
        projections/flats/darks/angles/metadata live.

        `handle` is the same object passed to matches().
        """

    @abstractmethod
    def parse_metadata(self, handle, path_map: PathMap) -> MetadataBundle:
        """
        Called after get_path_map(), also only for the selected plugin.
        Reads this vendor's metadata (voxel size, angles, operator
        name, etc.) into a MetadataBundle, sorted into tiers (see
        core/metadata.py).

        Does not load the full projections/flats/darks arrays; that
        happens in FormatBackend.build_data_handle(), using the paths
        returned from get_path_map(). Reads only small values: scalars,
        short arrays, attribute strings.
        """


# --- Registry -----------------------------------------------------------
#
# A plain in-memory registry: format ("hdf5"/"tiff") -> list of plugin
# instances. Plugins register via @register_plugin when their module is
# imported (see data_import/vendors/__init__.py).

_REGISTRY: dict[str, list[ImportPlugin]] = {"hdf5": [], "tiff": []}


def register_plugin(plugin_cls):
    """
    Decorator: instantiates a plugin class and adds it to the registry
    when its module is imported.

        @register_plugin
        class MyVendorPlugin(ImportPlugin):
            name = "my_vendor"
            format = "hdf5"
            ...
    """
    instance = plugin_cls()
    if instance.format not in _REGISTRY:
        raise ValueError(f"Unknown format '{instance.format}' for plugin {plugin_cls}")
    _REGISTRY[instance.format].append(instance)
    return plugin_cls


def get_plugins(format: Literal["hdf5", "tiff"]) -> list[ImportPlugin]:
    """Every currently-registered plugin for the given format."""
    return list(_REGISTRY[format])


def clear_registry():
    """Empty the registry. Used by tests."""
    _REGISTRY["hdf5"].clear()
    _REGISTRY["tiff"].clear()
