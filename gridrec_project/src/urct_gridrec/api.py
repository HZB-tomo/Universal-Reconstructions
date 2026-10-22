"""Public entry point: gridrec()."""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from urct_gridrec.core import Plan, make_plan, reconstruct_stack


def _backend(device: str):
    if device == "cpu":
        from urct_gridrec.cpu import CpuBackend

        return CpuBackend()
    if device == "reference":
        from urct_gridrec.reference import ReferenceBackend

        return ReferenceBackend()
    if device == "gpu":
        from urct_gridrec.gpu import GpuBackend

        return GpuBackend()
    raise ValueError(f"unknown device {device!r}; use 'cpu', 'gpu', 'auto' or 'reference'")


def gpu_available() -> bool:
    try:
        from urct_gridrec.gpu import gpu_available as check

        return check()
    except Exception:  # noqa: BLE001 -- numba/cupy missing or no driver: simply no GPU
        return False


def gridrec(
    sinograms: np.ndarray,
    angles: np.ndarray,
    center: Union[float, np.ndarray],
    *,
    device: str = "auto",
    filter_name: str = "hann",
    oversampling: float = 1.25,
    kernel_width: int = 6,
    apply_log: bool = True,
    ratio: Optional[float] = 1.0,
    plan: Optional[Plan] = None,
    timings: Optional[dict] = None,
) -> np.ndarray:
    """
    Reconstruct a stack of sinograms with gridrec.

    sinograms: (n_angles, n_slices, width), transmission values (or
        attenuation with apply_log=False). A (n_angles, width) sinogram
        returns one (width, width) image.
    angles: (n_angles,) in degrees.
    center: rotation axis position in detector columns; one value for all
        slices or one per slice (tilt).
    device: "cpu", "gpu", "auto" (GPU if there is one) or "reference"
        (slow float64 NumPy, for checking).
    filter_name: "ramp", "hann", "hamming", "cosine" or "shepp-logan".
    oversampling, kernel_width: gridding accuracy/speed. oversampling is the padded
        projection length (and grid size) as a multiple of the width; the defaults
        (1.25, 6) put the gridding error near 3e-4 of the image, far below the
        reconstruction's own error, and pad about as much as FBP does. Larger values
        (2.0) are more accurate but cost more time and memory.
    ratio: radius of the circular field of view as a fraction of the width
        (outside is zeroed, like the FBP backends); None keeps everything.
    plan: a make_plan() result to reuse across calls of the same width.
    timings: pass a dict to get seconds per stage added into it (profiling).

    Returns float32 (n_slices, width, width), axes (y, x), scaled like the
    main project's FBP backends. The input array is not modified.
    """
    sino = np.asarray(sinograms)
    single = sino.ndim == 2
    if single:
        sino = sino[:, None, :]
    n_angles, n_slices, width = sino.shape
    angles = np.asarray(angles, dtype=np.float64)
    if angles.shape != (n_angles,):
        raise ValueError(f"{angles.size} angles for {n_angles} projections")
    centers = np.broadcast_to(np.asarray(center, dtype=np.float64), (n_slices,))

    if device == "auto":
        device = "gpu" if gpu_available() else "cpu"
    if plan is None:
        plan = make_plan(width, oversampling, kernel_width, filter_name)
    image = reconstruct_stack(sino, np.radians(angles), centers, plan, _backend(device), ratio, timings, apply_log)
    return image[0] if single else image
