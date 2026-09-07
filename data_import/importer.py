"""
DataImporter: turns a file path into a ready ProjectObject. Does not
know how to read hdf5/tiff (backend.py) or where vendor data lives
(each plugin, in plugin.py and vendors/); it coordinates those pieces.

  detect(path, format)
      Identifies which plugin should handle a file, without loading
      data.

  import_project(path, format, storage_mode)
      detect -> ask the plugin where things are and what the metadata
      means -> ask the backend to build the data handles -> assemble
      and return a ProjectObject.
"""
from __future__ import annotations

from typing import Literal, Optional

from data_import.backend import Hdf5Backend, TiffBackend
from data_import.plugin import ImportPlugin, get_plugins
from data_import.errors import NoVendorMatchError, AmbiguousVendorMatchError, ProjectImportError
from core.project import ProjectObject
from core.known_fields import reconcile

# One backend instance per format, shared by every import call.
_BACKENDS = {
    "hdf5": Hdf5Backend(),
    "tiff": TiffBackend(),
}


class DataImporter:
    """
        importer = DataImporter()
        project = importer.import_project("my_scan.h5", format="hdf5", storage_mode="ram")
    """

    def detect(self, path, format: Literal["hdf5", "tiff"], opened=None) -> ImportPlugin:
        """
        Identify which registered plugin should handle this file,
        without loading data.

        1. Ask every registered plugin of this format whether its
           signature matches.
        2. If nothing matched, raise NoVendorMatchError with detail on
           why each plugin failed.
        3. Among matching plugins, find the highest specificity score.
        4. If more than one plugin ties at that top score, raise
           AmbiguousVendorMatchError.
        5. Otherwise, return the winning plugin.

        `opened` lets a caller pass an already-open handle to avoid
        opening the file twice; if omitted, this method opens and
        closes the file itself.
        """
        backend = _BACKENDS[format]
        we_opened_it = opened is None
        handle = opened if opened is not None else backend.open(path)
        try:
            candidates = get_plugins(format)
            results = [(plugin, plugin.matches(handle)) for plugin in candidates]
            matched = [(plugin, result) for plugin, result in results if result.matched]

            if not matched:
                raise NoVendorMatchError(
                    details=[(plugin.name, result.failed) for plugin, result in results]
                )

            matched.sort(key=lambda pr: pr[0].signature.specificity, reverse=True)
            top_score = matched[0][0].signature.specificity
            tied_for_top = [plugin for plugin, _ in matched if plugin.signature.specificity == top_score]

            if len(tied_for_top) > 1:
                raise AmbiguousVendorMatchError(candidates=[p.name for p in tied_for_top])

            return matched[0][0]
        finally:
            if we_opened_it:
                backend.close(handle)

    def import_project(
        self,
        path,
        format: Literal["hdf5", "tiff"],
        storage_mode: Literal["ram", "disk"] = "ram",
        plugin: Optional[ImportPlugin] = None,
    ) -> ProjectObject:
        """
        Returns a ready ProjectObject, or raises one of the errors in
        errors.py if the file can't be identified or is missing
        mandatory data.

        Args:
            path: file path (hdf5) or folder path (tiff).
            format: "hdf5" or "tiff".
            storage_mode: "ram" or "disk". "disk" is only valid for
                hdf5; requesting it for tiff raises ValueError.
            plugin: normally None, triggering automatic detect(). Pass
                an explicit plugin instance to skip detection.
        """
        backend = _BACKENDS[format]
        if format == "tiff" and storage_mode == "disk":
            raise ValueError("TIFF projects are always RAM-backed; use storage_mode='ram'.")

        opened = backend.open(path)

        selected_plugin = plugin or self.detect(path, format, opened=opened)

        path_map = selected_plugin.get_path_map(opened)
        metadata = selected_plugin.parse_metadata(opened, path_map)

        # Catches any field the plugin never mentioned, independent of
        # whether it called declare_missing() itself.
        reconcile(metadata)

        missing_hard = metadata.missing_hard()
        if missing_hard:
            backend.close(opened)
            raise ProjectImportError(missing_hard)

        data_handles = {}
        if path_map.projections:
            data_handles["projections"] = backend.build_data_handle(
                opened, path_map.projections, storage_mode
            )
        if path_map.flats:
            data_handles["flats"] = backend.build_data_handle(
                opened, path_map.flats, storage_mode
            )
        if path_map.darks:
            data_handles["darks"] = backend.build_data_handle(
                opened, path_map.darks, storage_mode
            )

        project = ProjectObject(
            format=format,
            storage_mode=storage_mode,
            metadata=metadata,
            data_handles=data_handles,
            vendor_plugin_name=selected_plugin.name,
            source_info={"path_map": path_map, "source_path": str(path)},
        )

        # In disk mode, the open h5py.File must stay open for as long as
        # the project's disk-backed handles are used; ProjectObject owns
        # that via close(). Otherwise, close now.
        if storage_mode == "disk":
            project._open_resource = opened
        else:
            backend.close(opened)

        return project
