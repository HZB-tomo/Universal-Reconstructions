"""
A pre-flight check for whether reconstructing a full volume is likely
to fit in available RAM, before starting a run that could otherwise
hang or thrash the machine partway through. See doc/GUI_ARCHITECTURE.md.

reconstruct()/postprocess()/export() all take and return a full
in-memory array; there is no streaming path that reconstructs and
writes a volume slice by slice without holding the whole thing in RAM.
Until that exists, this is the only safeguard against a project that's
simply too big.
"""
from __future__ import annotations

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
    The reconstructed volume's size in bytes, before any dtype
    conversion in postprocess(). Assumes a square (width, width)
    output image per slice -- true for every backend currently
    registered (algotom_fbp).
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
