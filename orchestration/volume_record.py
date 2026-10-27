"""
The settings a reconstruction contributes to its volume file (the record in
/entry/reconstruction, see reconstruction/volume_file.py): algorithm, filter,
iterations, device, rows, preprocessing and where the data came from. Derived
from the ProjectConfig so that the pipeline, the batch runner and the GUI's
"save reconstruction" all describe a volume the same way.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from orchestration.config import ProjectConfig
from reconstruction import get_backend, resolve_device


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_settings(config: ProjectConfig) -> dict:
    """Entries for the settings dict that volume_file.RECORD_KEYS files into the record."""
    backend = get_backend(config.reconstruction_backend)
    options = config.backend_options()
    out = {"backend": config.reconstruction_backend, "dtype": config.dtype}

    filter_name = options.pop("filter_name", None) or (backend.default_filter if backend.filters else None)
    if filter_name:
        out["filter_name"] = filter_name
    if backend.is_iterative and "num_iter" in options:
        out["num_iter"] = options.pop("num_iter")
    if options:
        out["algorithm_options"] = {k: v for k, v in options.items() if isinstance(v, (int, float, str, bool))}

    out["device_requested"] = config.device or "auto"
    try:
        out["device_used"] = resolve_device(config.reconstruction_backend, config.device)
    except Exception:       # a backend whose packages are missing cannot say; the run itself fails later
        pass
    if isinstance(config.rows, slice):
        out["rows"] = [config.rows.start or 0, config.rows.stop] if config.rows.stop is not None else None
    elif config.rows is not None:
        out["rows"] = [int(config.rows[0]), int(config.rows[1])]
    out["preprocessing"] = preprocessing_record(config)
    if config.path:
        out["source_file"] = config.path
    return {k: v for k, v in out.items() if v is not None}


def preprocessing_record(config: ProjectConfig) -> dict:
    record = {
        "normalization": "none (projections already corrected)" if config.projections_normalized else "flat/dark",
        "zinger_removal": config.zinger_removal,
        "ring_correction": config.ring_correction,
    }
    if config.ring_correction:
        record["ring_method"] = config.ring_method
        if config.ring_options:
            record["ring_options"] = _plain(config.ring_options)
    if config.phase_retrieval:
        record["phase_retrieval"] = _plain({
            "method": config.phase_method, "delta_over_beta": config.phase_delta_beta,
            "distance_mm": config.phase_distance_mm, "energy_kev": config.phase_energy_kev,
            "pixel_um": config.phase_pixel_um, **config.phase_options,
        })
    if config.beam_hardening and config.bh_coefficients is not None:
        record["beam_hardening"] = _plain({"coefficients": list(config.bh_coefficients), **config.bh_options})
    return record


def _plain(options: dict) -> dict:
    return {k: v for k, v in options.items() if v is not None and isinstance(v, (int, float, str, bool, list))}


class RunTimer:
    """Start/finish times and speed of a run, as settings entries."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self.started = now_iso()

    def entries(self, n_slices: int) -> dict:
        seconds = max(time.perf_counter() - self._t0, 1e-9)
        return {
            "started": self.started, "finished": now_iso(), "duration_s": round(seconds, 3),
            "n_slices": int(n_slices), "speed_slices_per_s": round(n_slices / seconds, 3),
        }
