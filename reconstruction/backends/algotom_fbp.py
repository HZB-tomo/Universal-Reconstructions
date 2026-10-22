"""
Parallel-beam FBP reconstruction backend wrapping
algotom.rec.reconstruction.fbp_reconstruction.

Requires the optional `algotom` package, imported lazily. Raises
ImportError with install instructions if it is not installed.

algotom's fbp_reconstruction() accepts a whole (n_angles, n_slices,
width) stack in ONE call: it filters the slices on several threads and,
on a GPU, backprojects all of them in a single kernel launch. Calling
it once per slice instead (what this backend used to do) rebuilds the
ramp filter every time and pays a host<->device copy and a kernel launch
per slice, which is what made the GPU path slow. So:

GPU:
- all slices share one center (no tilt): one batched call, split only as
  far as free GPU memory requires (see plan_batches());
- slices with different centers (tilt correction): one call per slice,
  but the ramp filter window is built once and reused.
CPU: algotom's own backprojection is not used at all -- it has a data
race that makes results wrong and different on every run; see
_fbp_cpu.py for the race-free replacement.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from reconstruction.backend import Geometry, ReconstructionBackend, register_backend
from reconstruction.backends._algotom_common import algotom_reconstruction

# Share of the GPU's free memory one batch may use: the filtered
# sinograms and the reconstructed images both live there at once, and
# numba/the driver need headroom.
_GPU_MEMORY_FRACTION = 0.5


def plan_batches(n_slices: int, batch_size: int) -> list[slice]:
    """Consecutive slice ranges of at most `batch_size` covering n_slices."""
    batch_size = max(1, batch_size)
    return [slice(start, min(start + batch_size, n_slices)) for start in range(0, n_slices, batch_size)]


def gpu_batch_size(n_angles: int, ncol: int, free_bytes: Optional[int]) -> int:
    """How many slices fit in one GPU batch: per slice, the filtered
    sinogram (n_angles x padded width) plus the (ncol x ncol) image,
    float32, doubled for working copies. Unknown free memory -> 16."""
    if free_bytes is None:
        return 16
    per_slice = (n_angles * (ncol + 300) + ncol * ncol) * 4 * 2
    return max(1, int(free_bytes * _GPU_MEMORY_FRACTION // per_slice))


def protect_input(sinograms: np.ndarray, options: dict) -> np.ndarray:
    """algotom's reconstructions replace every sinogram value <= 0 with 1
    IN PLACE before taking -log (apply_log=True, the default), silently
    changing the caller's array. Hand it a copy when that would happen."""
    if options.get("apply_log", True) and np.any(sinograms <= 0.0):
        return sinograms.copy()
    return sinograms


def gpu_free_bytes(rec) -> Optional[int]:
    try:
        return int(rec.cuda.current_context().get_memory_info()[0])
    except Exception:  # noqa: BLE001 -- no/odd CUDA state: fall back to the default batch size
        return None


@register_backend
class AlgotomFbpBackend(ReconstructionBackend):
    name = "algotom_fbp"
    supported_geometries = {"parallel"}

    def reconstruct(
        self,
        sinograms: np.ndarray,
        angles: np.ndarray,
        center,
        geometry: Geometry,
        device: Optional[str] = None,
        **options,
    ) -> np.ndarray:
        rec = algotom_reconstruction(self.name)

        gpu = True if device is None else (device == "gpu")
        if gpu and not rec.cuda.is_available():
            gpu = False  # algotom would warn and fall back anyway; decide once, here

        angles_rad = np.radians(angles)
        n_angles, n_slices, ncol = sinograms.shape
        centers = np.broadcast_to(np.asarray(center, dtype=float), (n_slices,))

        if not gpu:
            # algotom's CPU backprojection has a data race (see _fbp_cpu.py);
            # use the race-free kernel, which handles per-slice centers itself.
            from reconstruction.backends._fbp_cpu import fbp_stack

            return fbp_stack(rec, sinograms, centers, angles_rad, **options)

        sinograms = protect_input(sinograms, options)
        if np.all(centers == centers[0]):
            batch = gpu_batch_size(n_angles, ncol, gpu_free_bytes(rec))
            parts = []
            for chunk in plan_batches(n_slices, batch):
                out = rec.fbp_reconstruction(
                    sinograms[:, chunk, :], float(centers[0]), angles=angles_rad, gpu=True, **options,
                )
                parts.append(np.ascontiguousarray(np.moveaxis(out, 1, 0)))  # (ncol, n, ncol) -> (n, ncol, ncol)
            return parts[0] if len(parts) == 1 else np.concatenate(parts)

        # Different center per slice (tilt): the ramp filter does not
        # depend on the center, so build it once instead of per call.
        if "ramp_win" not in options:
            pad = options.get("pad")
            pad = min(int(0.15 * ncol), 150) if pad is None else pad
            options["ramp_win"] = rec.make_2d_ramp_window(n_angles, ncol + 2 * pad, options.get("filter_name", "hann"))
        images = [
            rec.fbp_reconstruction(sinograms[:, i, :], float(centers[i]), angles=angles_rad, gpu=True, **options)
            for i in range(n_slices)
        ]
        return np.stack(images)

    def actual_device(self, device: Optional[str] = None) -> str:
        wants_gpu = True if device is None else (device == "gpu")
        if not wants_gpu:
            return "cpu"
        # Mirrors fbp_reconstruction()'s own check exactly (it's
        # literally numba.cuda re-exported), so this reports the truth
        # rather than a separate guess that could disagree with it.
        return "gpu" if algotom_reconstruction(self.name).cuda.is_available() else "cpu"
