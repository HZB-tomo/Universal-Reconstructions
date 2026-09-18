"""
PipelineRun: drives one project through the full pipeline (import,
completion, preprocessing, calibration, reconstruction, postprocess,
export), configured entirely by a ProjectConfig.

Every stage is independently callable (a GUI can call one at a time and
inspect state between them) and run_all() chains all seven for the
common "just run it" case. No stage ever blocks on input: with
interactive=False (the default, and always the case in batch use),
missing information raises PipelineStageError; with interactive=True,
a single missing field is prompted for on stdin.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

import completion
from calibration import centers_for_rows, find_center, find_tilt
from core.data_interface import DataInterface
from core.project import ProjectObject
from data_import.errors import AmbiguousVendorMatchError, NoVendorMatchError, ProjectImportError
from data_import.importer import DataImporter
from data_import.plugin import get_plugin
from preprocessing import build_reference, normalize_rows, process_volume, remove_ring_artifacts, remove_zinger
from reconstruction import Geometry, StreamingVolumeWriter, reconstruct, save_volume, to_uint16

from orchestration import ram_check
from orchestration.config import PipelineStageError, ProjectConfig


@dataclass
class PipelineResult:
    project_id: str
    config: ProjectConfig
    success: bool
    volume_path: Optional[str] = None
    error: Optional[PipelineStageError] = None
    log: list = field(default_factory=list)


class PipelineRun:
    def __init__(self, config: ProjectConfig, interactive: bool = False):
        self.config = config
        self.interactive = interactive
        self.project_id = config.resolved_project_id()

        self.project: Optional[ProjectObject] = None
        self.reference = None
        self.normalized: Optional[np.ndarray] = None
        self.centers = None
        self.volume: Optional[np.ndarray] = None

        self.settings: dict = {}
        self.log: list = []

    def _log(self, message: str) -> None:
        self.log.append(message)

    # -- stages -----------------------------------------------------------

    def import_data(self) -> ProjectObject:
        try:
            plugin = get_plugin(self.config.format, self.config.plugin_name) if self.config.plugin_name else None
            self.project = DataImporter().import_project(
                self.config.path, format=self.config.format,
                storage_mode=self.config.storage_mode, plugin=plugin,
            )
        except (NoVendorMatchError, AmbiguousVendorMatchError, ProjectImportError, ValueError) as e:
            raise PipelineStageError("import", str(e)) from e

        self._log(f"imported via {self.project.vendor_plugin_name}")
        return self.project

    def complete(self) -> None:
        completion.run(self.project)
        for key, value in self.config.completion_answers.items():
            completion.provide(self.project, key, value)

        still_missing = self.project.missing_soft_fields()
        if still_missing:
            if self.interactive:
                for key in still_missing:
                    value = input(f"'{key}' is required and could not be resolved. Enter a value: ")
                    completion.provide(self.project, key, value)
            else:
                raise PipelineStageError("complete", f"missing required fields: {still_missing}", fields=still_missing)

        self._log("completion resolved")

    def preprocess(self) -> np.ndarray:
        self.reference = build_reference(
            self.project, reducer=self.config.reference_reducer, dark_offset=self.config.dark_offset,
        )
        rows = self.config.rows if self.config.rows is not None else slice(None)
        self.normalized = normalize_rows(self.project, rows, self.reference)

        if self.config.zinger_removal:
            self.normalized = process_volume(
                self.normalized, remove_zinger,
                self.config.zinger_threshold if self.config.zinger_threshold is not None else 0.08,
                size=self.config.zinger_size if self.config.zinger_size is not None else 2,
            )
        if self.config.ring_correction:
            self.normalized = process_volume(
                self.normalized, remove_ring_artifacts,
                method=self.config.ring_method, **self.config.ring_options,
            )

        self._log(f"preprocessed {self.normalized.shape[1]} rows")
        return self.normalized

    def calibrate(self) -> None:
        n_rows = self.normalized.shape[1]
        reference_row = (n_rows - 1) / 2.0
        middle_row = int(round(reference_row))

        if self.config.center is not None:
            center = self.config.center
        else:
            center = find_center(self.normalized[:, middle_row, :], backend=self.config.calibration_backend)

        if self.config.tilt_deg is not None:
            tilt_deg = self.config.tilt_deg
        else:
            tilt_kwargs = {
                k: v for k, v in {
                    "num_rows": self.config.tilt_num_rows,
                    "margin": self.config.tilt_margin,
                    "outlier_tolerance": self.config.tilt_outlier_tolerance,
                }.items() if v is not None
            }
            tilt_result = find_tilt(
                self.normalized, backend=self.config.calibration_backend, center=center, **tilt_kwargs,
            )
            tilt_deg = tilt_result.tilt_deg

        self.project.metadata.set_computed("center_of_rotation", center)
        self.project.metadata.set_computed("tilt_angle", tilt_deg)
        self.settings.update({
            "center": center, "tilt_deg": tilt_deg, "calibration_backend": self.config.calibration_backend,
        })

        self.centers = center if tilt_deg == 0.0 else centers_for_rows(
            tilt_deg, center, reference_row, np.arange(n_rows),
        )

        self._log(f"calibrated: center={center}, tilt_deg={tilt_deg}")

    def reconstruct(self) -> np.ndarray:
        # geometry_type and the cone-beam/pixel-size fields are read from
        # project.metadata, not from self.config: completion already
        # resolved them (see ProjectConfig.reconstruction_backend's
        # comment), so this is the single place they're allowed to come
        # from, not a second, possibly-disagreeing source.
        di = DataInterface(self.project)
        geometry = Geometry(
            geometry_type=di.get_metadata("geometry_type"),
            source_to_detector_distance=di.get_metadata("source_to_detector_distance"),
            source_to_object_distance=di.get_metadata("source_to_object_distance"),
            detector_pixel_size=di.get_metadata("detector_pixel_size"),
        )
        angles = di.get_metadata("angles")

        self.volume = reconstruct(
            self.normalized, angles, self.centers, geometry=geometry,
            backend=self.config.reconstruction_backend, device=self.config.device,
            **self.config.reconstruction_options,
        )
        self.settings["backend"] = self.config.reconstruction_backend
        self._log(f"reconstructed volume {self.volume.shape}")
        return self.volume

    def postprocess(self) -> np.ndarray:
        if self.config.dtype == "uint16":
            self.volume, clip_record = to_uint16(
                self.volume, self.config.dtype_clip_min, self.config.dtype_clip_max,
            )
            self.settings["dtype_clip"] = clip_record

        if self.config.crop is not None:
            (z0, z1), (y0, y1), (x0, x1) = self.config.crop
            self.volume = self.volume[z0:z1, y0:y1, x0:x1]
            self.settings["crop"] = self.config.crop

        self._log(f"postprocessed to {self.volume.dtype}, shape {self.volume.shape}")
        return self.volume

    def export(self) -> str:
        template = self.config.export_path or self._default_export_path_template()
        path = self._resolve_export_path(template)

        settings = {
            **self.project.export_metadata(),
            **self.settings,
            **self.config.export_extra_settings,
        }
        save_volume(
            self.volume, path, format=self.config.export_format,
            dataset_path=self.config.export_dataset_path, settings=settings,
        )
        self.settings["export_path"] = path
        self._log(f"exported to {path}")
        return path

    def run_all(self) -> PipelineResult:
        try:
            self.import_data()
            self.complete()
            if self.config.stream:
                path = self._run_streaming()
            else:
                self.preprocess()
                self.calibrate()
                self.reconstruct()
                self.postprocess()
                path = self.export()
            return PipelineResult(
                project_id=self.project_id, config=self.config, success=True,
                volume_path=path, log=list(self.log),
            )
        except PipelineStageError as e:
            return PipelineResult(
                project_id=self.project_id, config=self.config, success=False,
                error=e, log=list(self.log),
            )

    # -- streaming (chunk-at-a-time) reconstruction --------------------------

    def _run_streaming(self) -> str:
        """
        The streaming counterpart to preprocess()+calibrate()+
        reconstruct()+postprocess()+export(): reconstructs and writes
        one chunk of consecutive rows (slices) at a time, via
        StreamingVolumeWriter, so only one chunk's raw/normalized/
        reconstructed data is ever in memory -- never the whole
        project's volume. Every registered reconstruction backend
        already parallelizes across the slices *within* one chunk
        internally (see reconstruction/backends/ -- algotom's own
        joblib/GPU-batched implementations), so the chunk itself is
        the unit of both memory bounding and parallelism; see
        orchestration.ram_check.estimate_chunk_rows().

        Two things must already be resolved before this can run, and
        this raises PipelineStageError rather than silently falling
        back to the eager path's own auto-find behavior, which would
        require loading the whole raw projection stack once anyway --
        exactly what streaming exists to avoid:

        - `config.center`: there's no safe default (an arbitrary
          center would corrupt every reconstructed slice), and the GUI
          already resolves this automatically (image center) the
          moment a project is opened, before any button is clicked --
          see AdjustmentPanel._refresh_from_session(). A bare
          ProjectConfig used directly (CLI/scripting) needs `center`
          set explicitly for streaming.
        - `config.dtype_clip_min`/`dtype_clip_max`, only if
          `dtype == "uint16"`: to_uint16() would otherwise default to
          the whole volume's own min/max, which isn't knowable until
          every chunk has been seen.

        `config.tilt_deg` is the one exception: if still None, this
        treats it as 0.0 (no tilt) rather than raising -- the same
        convention the GUI's own tilt display already uses (Basic
        settings shows 0.0 until Find Tilt has actually run), and a
        real, common, zero-cost default (unlike center, an assumed
        tilt of 0 doesn't corrupt anything if a project genuinely has
        none), not a discovery step.
        """
        if self.config.center is None:
            raise PipelineStageError(
                "calibrate",
                "streaming mode requires config.center to already be resolved (not None) -- "
                "an arbitrary default would corrupt every reconstructed slice, and auto-finding "
                "it here would mean loading the whole raw projection stack once, which is exactly "
                "what streaming exists to avoid. Run/find it first, or set it directly.",
            )
        if self.config.dtype == "uint16" and (self.config.dtype_clip_min is None or self.config.dtype_clip_max is None):
            raise PipelineStageError(
                "postprocess",
                "streaming mode needs dtype_clip_min and dtype_clip_max set explicitly for a uint16 "
                "export -- there's no whole-volume min/max to fall back to a chunk at a time, unlike "
                "the normal (eager) path. Set both, or export as float32.",
            )

        center = self.config.center
        tilt_deg = self.config.tilt_deg if self.config.tilt_deg is not None else 0.0

        di = DataInterface(self.project)
        _, n_rows, width = di.shape_of("projections")

        rows_slice = self.config.rows if self.config.rows is not None else slice(0, n_rows)
        rows_start, rows_stop, _ = rows_slice.indices(n_rows)
        total_rows = rows_stop - rows_start
        if total_rows <= 0:
            raise PipelineStageError("preprocess", f"config.rows={self.config.rows!r} selects no rows")
        # Relative to [rows_start, rows_stop), the same range calibrate()
        # would compute reference_row/centers_for_rows() against in the
        # eager path (self.normalized.shape[1] there is this same,
        # config.rows-restricted count) -- matched exactly here so
        # streaming and eager reconstruct identical values for the same
        # config, not just "close".
        reference_row = (total_rows - 1) / 2.0

        # crop's z-range is, like the eager path's postprocess()
        # (self.volume[z0:z1, ...]), relative to this same
        # config.rows-restricted range, not absolute detector rows --
        # applied here as a WRITE filter (skip writing rows outside it)
        # rather than a narrower read range, so the reference_row/
        # centers_for_rows() math above -- and therefore the actual
        # reconstructed values -- stays identical to the eager path
        # regardless of whether a crop is set. y/x crop, having nothing
        # to do with which rows exist, applies straightforwardly to
        # every chunk.
        if self.config.crop is not None:
            (z0, z1), (y0, y1), (x0, x1) = self.config.crop
            crop_yx: Optional[tuple] = (y0, y1, x0, x1)
            out_height, out_width = y1 - y0, x1 - x0
        else:
            z0, z1 = 0, total_rows
            crop_yx = None
            out_height, out_width = width, width
        out_total_rows = z1 - z0
        if out_total_rows <= 0:
            raise PipelineStageError("postprocess", f"config.crop={self.config.crop!r} selects no rows")

        self.reference = build_reference(
            self.project, reducer=self.config.reference_reducer, dark_offset=self.config.dark_offset,
        )
        geometry = Geometry(
            geometry_type=di.get_metadata("geometry_type"),
            source_to_detector_distance=di.get_metadata("source_to_detector_distance"),
            source_to_object_distance=di.get_metadata("source_to_object_distance"),
            detector_pixel_size=di.get_metadata("detector_pixel_size"),
        )
        angles = di.get_metadata("angles")

        self.project.metadata.set_computed("center_of_rotation", center)
        self.project.metadata.set_computed("tilt_angle", tilt_deg)
        self.settings.update({
            "center": center, "tilt_deg": tilt_deg, "calibration_backend": self.config.calibration_backend,
            "backend": self.config.reconstruction_backend,
        })
        if self.config.dtype == "uint16":
            self.settings["dtype_clip"] = {
                "clip_min": self.config.dtype_clip_min, "clip_max": self.config.dtype_clip_max,
                "source_dtype": "float32",
            }
        if self.config.crop is not None:
            self.settings["crop"] = self.config.crop

        chunk_rows = self.config.stream_chunk_rows or ram_check.estimate_chunk_rows(width, total_rows)

        template = self.config.export_path or self._default_export_path_template()
        path = self._resolve_export_path(template)
        export_settings = {
            **self.project.export_metadata(),
            **self.settings,
            **self.config.export_extra_settings,
        }
        out_dtype = np.uint16 if self.config.dtype == "uint16" else np.float32

        writer = StreamingVolumeWriter(
            path, format=self.config.export_format, total_slices=out_total_rows,
            height=out_height, width=out_width, dtype=out_dtype,
            dataset_path=self.config.export_dataset_path, settings=export_settings,
        )
        try:
            for chunk_start in range(rows_start, rows_stop, chunk_rows):
                chunk_stop = min(chunk_start + chunk_rows, rows_stop)
                rel_start, rel_stop = chunk_start - rows_start, chunk_stop - rows_start

                write_start, write_stop = max(rel_start, z0), min(rel_stop, z1)
                if write_start >= write_stop:
                    continue  # this whole chunk falls outside config.crop's z-range

                chunk = normalize_rows(self.project, slice(chunk_start, chunk_stop), self.reference)
                if self.config.zinger_removal:
                    chunk = process_volume(
                        chunk, remove_zinger,
                        self.config.zinger_threshold if self.config.zinger_threshold is not None else 0.08,
                        size=self.config.zinger_size if self.config.zinger_size is not None else 2,
                    )
                if self.config.ring_correction:
                    chunk = process_volume(
                        chunk, remove_ring_artifacts,
                        method=self.config.ring_method, **self.config.ring_options,
                    )

                chunk_centers = center if tilt_deg == 0.0 else centers_for_rows(
                    tilt_deg, center, reference_row, np.arange(rel_start, rel_stop),
                )
                chunk_volume = reconstruct(
                    chunk, angles, chunk_centers, geometry=geometry,
                    backend=self.config.reconstruction_backend, device=self.config.device,
                    **self.config.reconstruction_options,
                )

                if self.config.dtype == "uint16":
                    chunk_volume, _ = to_uint16(chunk_volume, self.config.dtype_clip_min, self.config.dtype_clip_max)
                if crop_yx is not None:
                    chunk_volume = chunk_volume[:, y0:y1, x0:x1]

                # Trim to just the part of this chunk config.crop's
                # z-range actually keeps, then place it at that range's
                # own 0-based offset -- see the docstring above.
                chunk_volume = chunk_volume[write_start - rel_start:write_stop - rel_start]
                writer.write_chunk(chunk_volume, z_offset=write_start - z0)
                self._log(f"streamed rows {chunk_start}:{chunk_stop} ({chunk_stop - chunk_start} slices)")
        finally:
            writer.close()

        self.settings["export_path"] = path
        self._log(f"streamed {out_total_rows} rows to {path} in chunks of up to {chunk_rows}")
        return path

    # -- export path resolution --------------------------------------------

    def _default_export_path_template(self) -> str:
        ext = "h5" if self.config.export_format == "hdf5" else None
        return f"{{project_id}}_volume.{ext}" if ext else "{project_id}_volume"

    def _resolve_export_path(self, template: str) -> str:
        """
        Fill in `template`'s {placeholder}s from self.settings plus
        project_id. A placeholder with no value resolves to "NAN"
        rather than raising, since a naming mistake shouldn't be able
        to fail an export after the expensive stages already ran. If
        the resolved path already exists, append "_1", "_2", ... until
        it doesn't, so two projects that both resolve to the same name
        (e.g. both missing the same field) never silently overwrite
        each other.
        """
        values = {**self.settings, "project_id": self.project_id}

        class _NanDict(dict):
            def __missing__(self, key):
                return "NAN"

        path = Path(template.format_map(_NanDict(values)))
        if not path.exists():
            return str(path)

        stem, suffix = path.stem, path.suffix
        n = 1
        candidate = path
        while candidate.exists():
            candidate = path.with_name(f"{stem}_{n}{suffix}")
            n += 1
        return str(candidate)
