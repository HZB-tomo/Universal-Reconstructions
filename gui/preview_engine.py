"""
PreviewEngine: single-slice previews with memory. Qt-free.

Trying out centers of rotation means reconstructing the same slice
again and again with a slightly different center. Almost everything in
that is repeated work, so the engine keeps

- the flat/dark reference and the normalized (and zinger/ring
  corrected) sinogram of the slice, which don't depend on the center;
- every reconstructed preview image, keyed by everything that changes
  the picture, so going back to a center already tried (mouse wheel,
  a candidate list) shows instantly and prefetch() can fill the
  neighbours in the background.

Nothing here touches the project's metadata or the GUI's PipelineRun,
so it is safe to call from a background thread while the user keeps
working.
"""
from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from typing import Callable, Iterable, Optional

import numpy as np

from core.data_interface import DataInterface
from orchestration import ProjectConfig
from orchestration.geometry import geometry_from_project
from orchestration.preprocess import preprocess_rows
from preprocessing import build_reference
from reconstruction import Geometry, reconstruct, reconstruct_rotated

DEFAULT_MAX_BYTES = 512 * 1024 ** 2
_MAX_NORMALIZED_ROWS = 8


def _dict_key(d: dict) -> str:
    return repr(sorted((str(k), repr(v)) for k, v in (d or {}).items()))


def preprocessing_key(config: ProjectConfig) -> tuple:
    """Settings that change the normalized sinogram of a row."""
    return (
        config.reference_reducer, config.dark_offset, config.projections_normalized,
        config.zinger_removal, config.zinger_threshold, config.zinger_size,
        config.ring_correction, config.ring_method, _dict_key(config.ring_options),
        config.phase_retrieval, config.phase_method, config.phase_delta_beta, config.phase_distance_mm,
        config.phase_energy_kev, config.phase_pixel_um, _dict_key(config.phase_options),
        config.beam_hardening, config.bh_coefficients, _dict_key(config.bh_options),
    )


def reconstruction_key(config: ProjectConfig) -> tuple:
    """Settings, other than the row and the center, that change the picture."""
    return (
        preprocessing_key(config), config.reconstruction_backend, config.device,
        _dict_key(config.backend_options()), config.rotate_deg or 0.0, _dict_key(config.completion_answers),
    )


class PreviewEngine:
    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES):
        self.max_bytes = max_bytes
        self._images: OrderedDict = OrderedDict()
        self._bytes = 0
        self._normalized: OrderedDict = OrderedDict()
        self._references: dict = {}
        self._lock = threading.RLock()  # guards the dicts and the one-time sinogram preparation
        self.reconstructions = 0  # how many images were actually computed (for tests / diagnostics)

    # -- cache ---------------------------------------------------------------

    @staticmethod
    def image_key(project, config: ProjectConfig, row: int, center: float) -> tuple:
        return (id(project), int(row), round(float(center), 4), reconstruction_key(config))

    def cached(self, project, config: ProjectConfig, row: int, center: float) -> Optional[np.ndarray]:
        key = self.image_key(project, config, row, center)
        with self._lock:
            image = self._images.get(key)
            if image is not None:
                self._images.move_to_end(key)
            return image

    def _store(self, key: tuple, image: np.ndarray) -> None:
        with self._lock:
            if key in self._images:
                self._bytes -= self._images[key].nbytes
            self._images[key] = image
            self._images.move_to_end(key)
            self._bytes += image.nbytes
            while self._bytes > self.max_bytes and len(self._images) > 1:
                _, old = self._images.popitem(last=False)
                self._bytes -= old.nbytes

    def clear(self) -> None:
        with self._lock:
            self._images.clear()
            self._normalized.clear()
            self._references.clear()
            self._bytes = 0

    def __len__(self) -> int:
        return len(self._images)

    # -- computing -----------------------------------------------------------

    def _normalized_row(self, project, config: ProjectConfig, row: int) -> np.ndarray:
        """(n_angles, 1, width), normalized and corrected; built once per row/settings."""
        key = (id(project), int(row), preprocessing_key(config))
        with self._lock:
            hit = self._normalized.get(key)
            if hit is not None:
                self._normalized.move_to_end(key)
                return hit
            ref_key = (id(project), config.reference_reducer, config.dark_offset, config.projections_normalized)
            reference = self._references.get(ref_key)
            if reference is None:
                reference = build_reference(
                    project, reducer=config.reference_reducer, dark_offset=config.dark_offset,
                    already_normalized=config.projections_normalized,
                )
                self._references[ref_key] = reference
            normalized = preprocess_rows(project, config, reference, slice(row, row + 1))
            self._normalized[key] = normalized
            while len(self._normalized) > _MAX_NORMALIZED_ROWS:
                self._normalized.popitem(last=False)
            return normalized

    def compute(self, project, config: ProjectConfig, row: int, center: float) -> np.ndarray:
        """The reconstructed (height, width) slice `row` with rotation center
        `center`: from the cache if it was computed before, else computed
        now and remembered. `project` must already be completed."""
        hit = self.cached(project, config, row, center)
        if hit is not None:
            return hit
        normalized = self._normalized_row(project, config, row)
        di = DataInterface(project)
        geometry = geometry_from_project(project)
        if geometry.geometry_type == "cone":
            raise ValueError(
                "single-slice previews are not available for cone-beam data (every slice needs every detector row): "
                "use 'Reconstruct all slices'"
            )
        angles = di.get_metadata("angles")
        if config.rotate_deg:
            volume = reconstruct_rotated(
                normalized, angles, float(center), geometry=geometry, backend=config.reconstruction_backend,
                rotate_deg=config.rotate_deg, device=config.device, **config.backend_options(),
            )
        else:
            volume = reconstruct(
                normalized, angles, float(center), geometry=geometry, backend=config.reconstruction_backend,
                device=config.device, **config.backend_options(),
            )
        image = np.array(volume[0], dtype=np.float32)
        with self._lock:
            self.reconstructions += 1
        self._store(self.image_key(project, config, row, center), image)
        return image

    def prefetch(
        self, project, config: ProjectConfig, row: int, centers: Iterable[float],
        cancelled: Callable[[], bool] = lambda: False,
    ) -> int:
        """Computes (and remembers) previews for `centers`, nearest first as
        given, stopping as soon as `cancelled()` says so. Works on a private
        copy of `config`, so the GUI can keep editing it. Returns how many
        images it computed."""
        config = copy.deepcopy(config)
        done = 0
        for center in centers:
            if cancelled():
                break
            if self.cached(project, config, row, center) is not None:
                continue
            self.compute(project, config, row, center)
            done += 1
        return done
