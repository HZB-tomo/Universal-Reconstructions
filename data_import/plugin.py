"""
This file has two jobs, kept together because they're small and tightly
related:

  1. Define what EVERY import plugin must implement (the ImportPlugin
     class below). This is "the contract" -- if you're writing support
     for a new instrument/vendor, this is the class you subclass.

  2. Provide a simple way for plugins to "sign up" so DataImporter can
     find them (register_plugin / get_plugins below). This is
     deliberately just a list in memory, not anything fancier -- there's
     no need for plugin discovery across separate packages yet. If we
     ever want third parties to ship plugins as a separate installable
     package, we'd add that discovery mechanism ON TOP of this registry
     without needing to change how plugins themselves are written.

Recall from our discussion: a plugin answers exactly three questions
about ONE vendor's file convention:
  - "Is this my file?"          -> handled by `signature` (see signature.py)
  - "If so, where's everything?" -> get_path_map()
  - "What does the metadata mean?" -> parse_metadata()

A plugin never opens a file itself (that's FormatBackend's job, see
backend.py) and never decides RAM vs disk (that's the user's choice,
passed through DataImporter, see importer.py). A plugin is pure
knowledge about ONE vendor's conventions, nothing else.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from data_import.signature import Signature, SignatureResult
from data_import.path_map import PathMap
from core.metadata import MetadataBundle


class ImportPlugin(ABC):
    """
    Base class every vendor plugin inherits from. To add support for a
    new instrument, you write ONE subclass with these three things
    filled in -- nothing else in the codebase needs to change.

    Class attributes to set on your subclass:
        name:      a short unique string identifying this plugin, e.g.
                   "zeiss_xradia". Used for logging and for
                   ProjectObject.vendor_plugin_name.
        format:    either "hdf5" or "tiff" -- which format backend this
                   plugin works with.
        signature: an instance of Hdf5Signature or TiffSignature (must
                   match `format`) describing how to recognize this
                   vendor's files. See signature.py.
    """

    name: str
    format: Literal["hdf5", "tiff"]
    signature: Signature

    def matches(self, target) -> SignatureResult:
        """
        Check whether `target` (an open h5py.File for hdf5 plugins, or a
        folder Path for tiff plugins) satisfies this plugin's signature.
        You normally don't need to override this -- it just delegates to
        self.signature.check(). DataImporter calls this on every
        registered plugin during detection.
        """
        return self.signature.check(target)

    @abstractmethod
    def get_path_map(self, handle) -> PathMap:
        """
        Called ONLY after this plugin has already been selected as the
        best match (see importer.py). Given the open handle, return a
        PathMap pointing to where projections/flats/darks/angles/
        metadata live for this vendor's convention.

        `handle` is the same object passed to matches() -- an open
        h5py.File, or a folder Path.
        """

    @abstractmethod
    def parse_metadata(self, handle, path_map: PathMap) -> MetadataBundle:
        """
        Called right after get_path_map(), also only for the selected
        plugin. Read whatever metadata this vendor's files contain (voxel
        size, angles, operator name, an existing center-of-rotation
        guess, etc.) and sort each one into the correct tier using a
        MetadataBundle (see core/metadata.py).

        This is where the vendor-specific JUDGMENT CALLS happen: only
        this plugin knows, for example, whether ITS vendor always
        includes voxel size (so it can be tier="optional" or even
        skipped) or sometimes omits it (so it should be
        tier="mandatory_soft" with declare_missing() when absent).

        Do NOT load the full projections/flats/darks arrays here -- that
        happens later, in FormatBackend.build_data_handle(), driven by
        the paths you returned from get_path_map(). This method should
        only read small things: scalar values, small arrays like an
        angle list, attribute strings.
        """


# --- Registry -----------------------------------------------------------
#
# A plain in-memory registry: a dict from format ("hdf5"/"tiff") to a
# list of plugin instances. Plugins add themselves here via the
# @register_plugin decorator when their module is imported (see
# data_import/vendors/__init__.py, which imports every vendor module so
# this happens automatically on startup).

_REGISTRY: dict[str, list[ImportPlugin]] = {"hdf5": [], "tiff": []}


def register_plugin(plugin_cls):
    """
    Decorator: put @register_plugin above a plugin class definition and
    it will automatically be instantiated and added to the registry as
    soon as its module is imported. Usage:

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
    """Return every currently-registered plugin for the given format.
    DataImporter calls this during detection to know which plugins to
    test against a given file."""
    return list(_REGISTRY[format])


def clear_registry():
    """Empty the registry completely. Exists mainly for tests, so each
    test can start from a known, clean state rather than accumulating
    plugins registered by previous tests."""
    _REGISTRY["hdf5"].clear()
    _REGISTRY["tiff"].clear()
