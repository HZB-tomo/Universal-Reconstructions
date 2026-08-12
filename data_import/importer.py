"""
This file defines DataImporter, the single entry point the rest of the
software calls to turn "a file path" into "a ready ProjectObject". It's
the orchestrator: it doesn't know HOW to read hdf5/tiff (that's
backend.py) and it doesn't know WHERE vendor data lives (that's each
plugin, in plugin.py + vendors/) -- it just coordinates those pieces in
the right order.

The two things DataImporter offers:

  detect(path, format)
      Figure out which plugin should handle this file, without actually
      importing anything. Useful on its own, e.g. for a GUI that wants
      to show "detected vendor: Zeiss Xradia" before the user commits to
      loading gigabytes of data.

  import_project(path, format, storage_mode)
      The full pipeline: detect -> ask the winning plugin where things
      are and what the metadata means -> ask the backend to actually
      build the data handles -> assemble and return a ProjectObject.

See core/project.py's ProjectObject docstring, and plugin.py's
ImportPlugin docstring, for how the pieces below fit together -- this
file is deliberately just "glue", so if you're trying to understand what
actually HAPPENS to a file during import, reading import_project() top
to bottom below should tell the whole story.
"""
from __future__ import annotations

from typing import Literal, Optional

from data_import.backend import Hdf5Backend, TiffBackend
from data_import.plugin import ImportPlugin, get_plugins
from data_import.errors import NoVendorMatchError, AmbiguousVendorMatchError, ProjectImportError
from core.project import ProjectObject

# One backend instance per format, shared by every import call. Backends
# hold no per-file state of their own (they're just "how to read this
# format" logic), so there's no need to create a new one each time.
_BACKENDS = {
    "hdf5": Hdf5Backend(),
    "tiff": TiffBackend(),
}


class DataImporter:
    """The orchestrator described above. Typically used as:

        importer = DataImporter()
        project = importer.import_project("my_scan.h5", format="hdf5", storage_mode="ram")
    """

    def detect(self, path, format: Literal["hdf5", "tiff"], opened=None) -> ImportPlugin:
        """
        Figure out which registered plugin should handle this file,
        WITHOUT loading any actual data.

        Steps:
          1. Ask every registered plugin of this format whether its
             signature matches (see ImportPlugin.matches() / signature.py).
          2. If nothing matched at all -> raise NoVendorMatchError, with
             full detail on why each plugin failed.
          3. Among the plugins that DID match, find the highest
             specificity score.
          4. If more than one plugin is tied at that top score -> raise
             AmbiguousVendorMatchError (we refuse to guess).
          5. Otherwise, return the single winning plugin.

        `opened` lets a caller (like import_project() below) pass an
        already-open handle to avoid opening the file twice. If not
        given, this method opens and closes the file itself.
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

            # Sort matched plugins by specificity, highest first, so the
            # most specific (most likely correct) plugin is matched[0].
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
        The full import pipeline. Returns a ready ProjectObject, or
        raises one of the errors in errors.py if the file can't be
        identified or is missing mandatory data.

        Args:
            path: file path (hdf5) or folder path (tiff) to import.
            format: "hdf5" or "tiff".
            storage_mode: "ram" or "disk". Disk mode is only meaningful
                for hdf5 -- requesting it for tiff raises ValueError,
                since tiff data is always fully loaded into RAM (see
                backend.py's TiffBackend docstring for why).
            plugin: normally left as None, which triggers automatic
                detect(). Pass an explicit plugin instance to skip
                detection entirely -- e.g. after the user manually
                resolved an AmbiguousVendorMatchError by picking one.
        """
        backend = _BACKENDS[format]
        if format == "tiff" and storage_mode == "disk":
            raise ValueError("TIFF projects are always RAM-backed; use storage_mode='ram'.")

        opened = backend.open(path)

        # Step 1: figure out which plugin understands this file, unless
        # the caller already told us which one to use.
        selected_plugin = plugin or self.detect(path, format, opened=opened)

        # Step 2: ask the winning plugin where things are, and what the
        # metadata means. Neither of these steps reads the big arrays --
        # just paths and small metadata values (see plugin.py).
        path_map = selected_plugin.get_path_map(opened)
        metadata = selected_plugin.parse_metadata(opened, path_map)

        # Step 3: if any mandatory_hard field is still missing after the
        # plugin did its best, this file isn't usable -- stop here
        # rather than handing back a broken ProjectObject.
        missing_hard = metadata.missing_hard()
        if missing_hard:
            backend.close(opened)
            raise ProjectImportError(missing_hard)

        # Step 4: now actually read the big arrays, via the backend, at
        # whichever paths the plugin pointed us to. Only build a handle
        # for each array that the plugin actually found a path for --
        # e.g. some vendors won't have separate flats/darks at all.
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

        # Step 5: assemble the final ProjectObject.
        project = ProjectObject(
            format=format,
            storage_mode=storage_mode,
            metadata=metadata,
            data_handles=data_handles,
            vendor_plugin_name=selected_plugin.name,
            source_info={"path_map": path_map, "source_path": str(path)},
        )

        # If we're in disk mode, the open h5py.File must stay open for
        # as long as the project's disk-backed handles are used -- hand
        # that responsibility to the ProjectObject itself (see its
        # close() method). Otherwise, we're done reading, so close now.
        if storage_mode == "disk":
            project._open_resource = opened
        else:
            backend.close(opened)

        return project
