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
from calibration import CenterScan, half_scan_slice, image_center, scan_centers, scan_from_curve, search_center
from calibration.metrics import get_metric
from reconstruction import Geometry, reconstruct, reconstruct_rotated

DEFAULT_MAX_BYTES = 512 * 1024 ** 2
_MAX_NORMALIZED_ROWS = 8
# Cone beam: the one slice that can be reconstructed from its own detector rows is the central one, the plane
# through the source perpendicular to the rotation axis (every ray in it stays in it). It stands in for `row`.
CENTRAL_PLANE = -1
AUTO_HALF_WINDOW = 0.1               # the quick estimate looks this fraction of the width either side of the middle
CONE_APPROXIMATE = (
    "cone beam: this method assumes parallel rays, so on the fan-beam sinogram of the central plane it is only "
    "approximate (the error grows with the cone angle); the image-based metrics are exact"
)


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
        preprocessing_key(config), config.reconstruction_backend, config.device, config.reconstruction_filter,
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
    def is_cone(project) -> bool:
        try:
            return DataInterface(project).get_metadata("geometry_type") == "cone"
        except Exception:  # noqa: BLE001 -- not completed yet: the geometry is not known, so not cone
            return False

    @classmethod
    def effective_row(cls, project, row: int) -> int:
        """The row a preview really reconstructs: `row` for parallel beam, the central plane for cone beam."""
        return CENTRAL_PLANE if cls.is_cone(project) else int(row)

    @classmethod
    def image_key(cls, project, config: ProjectConfig, row: int, center: float) -> tuple:
        return (id(project), cls.effective_row(project, row), round(float(center), 4), reconstruction_key(config))

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
            if row == CENTRAL_PLANE:
                n_rows = DataInterface(project).shape_of("projections")[1]
                lo, hi = (n_rows - 1) // 2, n_rows // 2 + 1       # one row, or the two around the plane (even count)
                block = preprocess_rows(project, config, reference, slice(lo, hi))
                normalized = block.mean(axis=1, keepdims=True, dtype=np.float32)
            else:
                normalized = preprocess_rows(project, config, reference, slice(row, row + 1))
            self._normalized[key] = normalized
            while len(self._normalized) > _MAX_NORMALIZED_ROWS:
                self._normalized.popitem(last=False)
            return normalized

    def compute(self, project, config: ProjectConfig, row: int, center: float) -> np.ndarray:
        """The reconstructed (height, width) slice `row` with rotation center
        `center`: from the cache if it was computed before, else computed
        now and remembered. `project` must already be completed."""
        config.resolve_backend(DataInterface(project).get_metadata("geometry_type"))     # nothing chosen: the default
        row = self.effective_row(project, row)
        hit = self.cached(project, config, row, center)
        if hit is not None:
            return hit
        normalized = self._normalized_row(project, config, row)
        di = DataInterface(project)
        geometry = geometry_from_project(project)
        angles = di.get_metadata("angles")
        options = config.backend_options()
        if row == CENTRAL_PLANE:
            options["size_z"] = 1      # one slice, at the height of the source
        if config.rotate_deg:
            volume = reconstruct_rotated(
                normalized, angles, float(center), geometry=geometry, backend=config.reconstruction_backend,
                rotate_deg=config.rotate_deg, device=config.device, **options,
            )
        else:
            volume = reconstruct(
                normalized, angles, float(center), geometry=geometry, backend=config.reconstruction_backend,
                device=config.device, **options,
            )
        image = np.array(volume[0], dtype=np.float32)
        with self._lock:
            self.reconstructions += 1
        self._store(self.image_key(project, config, row, center), image)
        return image

    def scan(
        self, project, config: ProjectConfig, row: int, metric: str, centers, progress=None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> CenterScan:
        """The curve of `metric` over the trial `centers` for the slice `row` (for cone beam the central
        plane). Slice metrics go through compute(), so every picture they need is remembered: scrubbing the
        center afterwards is instant. Works on a private copy of `config`. Raises calibration.ScanCancelled
        when `cancelled()` turns true."""
        config = copy.deepcopy(config)
        row = self.effective_row(project, row)
        sinogram = angles = None
        reconstruct_at = None
        approximate = ""
        if get_metric(metric).kind == "sinogram":
            sinogram = self._normalized_row(project, config, row)[:, 0, :]
            angles = DataInterface(project).get_metadata("angles")
            if row == CENTRAL_PLANE:
                approximate = CONE_APPROXIMATE
        else:
            reconstruct_at = lambda center: self.compute(project, config, row, center)
        return scan_centers(
            metric, centers, sinogram=sinogram, angles=angles, reconstruct=reconstruct_at, progress=progress,
            cancelled=cancelled, approximate=approximate,
        )

    def auto_center(
        self, project, config: ProjectConfig, row: int, window: Optional[tuple] = None, progress=None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> tuple:
        """
        The quick estimate: Vo's search (calibration.search_center(), coarse over every column of the window,
        then refined to 0.25 px) over `window`, given as shifts from the detector middle: by default a tenth of
        the width either side, since a real center is within a few percent of the middle. Returns (CenterSearch,
        CenterScan, window): the search, the curve the search itself evaluated on the way (shown, not recomputed;
        None if the backend keeps none) with the search's own refined answer as its best, and the window it
        covers as shifts from the detector middle. `search.at_edge` says the answer sits at the edge of the window.
        """
        config = copy.deepcopy(config)
        row = self.effective_row(project, row)
        sinogram = self._normalized_row(project, config, row)[:, 0, :]
        angles = DataInterface(project).get_metadata("angles")
        width = sinogram.shape[1]
        if window is None:
            window = (-AUTO_HALF_WINDOW * width, AUTO_HALF_WINDOW * width)
        middle = image_center(width)
        search = search_center(
            sinogram[half_scan_slice(angles)], backend=config.calibration_backend,
            start=max(0.0, middle + window[0]), stop=min(width - 1.0, middle + window[1]),
            n_candidates=1,       # refining a candidate at full resolution is most of the cost; the curve shows any rival
        )
        if search.curve is None:
            return search, None, None
        centers, values = search.curve
        scan = scan_from_curve(
            "vo", centers, values, best=search.best, approximate=CONE_APPROXIMATE if row == CENTRAL_PLANE else "",
        )
        return search, scan, (float(centers[0]) - middle, float(centers[-1]) - middle)

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
