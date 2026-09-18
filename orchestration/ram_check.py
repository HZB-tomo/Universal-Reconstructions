"""
RAM estimation for reconstruction: a pre-flight check for the eager
(whole-volume) path, and chunk sizing for the streaming path. Lives in
orchestration/, not gui/, since both the GUI and the CLI/batch path
need it -- see PipelineRun._run_streaming() in pipeline.py.
"""
from __future__ import annotations

import os

import psutil

# The float32 output array is not the only thing in memory during a
# reconstruction: the backend's own intermediate buffers, a
# postprocess() dtype conversion (which briefly holds both the old and
# new array at once), and normal allocator fragmentation all add to
# the peak. A rough safety margin, not a measured figure.
SAFETY_FACTOR = 2.5

BYTES_PER_FLOAT32 = 4


def estimate_volume_bytes(n_rows: int, width: int) -> int:
    """
    A reconstructed volume's (or one chunk's) size in bytes, before
    any dtype conversion in postprocess(). Assumes a square (width,
    width) output image per slice -- true for every backend currently
    registered.
    """
    return n_rows * width * width * BYTES_PER_FLOAT32


def fits_in_available_ram(required_bytes: int) -> bool:
    return psutil.virtual_memory().available >= required_bytes * SAFETY_FACTOR


def describe(num_bytes: int) -> str:
    """Human-readable size, e.g. "2.3 GB"."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def estimate_chunk_rows(width: int, max_rows: int) -> int:
    """
    How many rows (slices) a streaming reconstruction should process
    per chunk: every registered reconstruction backend already
    parallelizes across the slices in a single reconstruct() call
    internally (algotom's own joblib/GPU-batched implementations --
    confirmed directly by reading their source, not assumed), so a
    chunk of just 1 row wastes that -- it forces every call to run on
    a single core/one slice at a time. Sized to keep every CPU core
    busy (os.cpu_count()) unless that wouldn't fit in available RAM
    (or there simply aren't that many rows left to give it), in which
    case RAM/the remaining row count wins -- never exceed either.
    Clamped to at least 1, so a single very large row can still be
    processed rather than raising.
    """
    if max_rows <= 0:
        return 0
    per_row_bytes = estimate_volume_bytes(1, width) * SAFETY_FACTOR
    ram_limited_rows = max(1, int(psutil.virtual_memory().available // per_row_bytes))
    cpu_friendly_rows = max(1, os.cpu_count() or 1)
    return max(1, min(cpu_friendly_rows, ram_limited_rows, max_rows))
