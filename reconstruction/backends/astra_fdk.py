"""
Cone-beam FDK reconstruction backend on the GPU, wrapping the ASTRA Toolbox's
`FDK_CUDA` (pip install astra-toolbox; it brings its own CUDA runtime, so only an
NVIDIA driver is needed).

The same scan model, conventions and result as `rtk_fdk` (see its docstring and
doc/CONE_BEAM.md): circular orbit, flat detector, rotation axis vertical with
detector rows along it, at angle theta the detector's column axis is
u = (cos theta, -sin theta) and the ray direction r = (sin theta, cos theta), the
source sits at -SOD * r and the detector plane at (SDD - SOD) * r. `center` is
the detector column onto which the rotation axis projects. The result is
(slices, y, x) with cubic voxels of the size a detector pixel has at the axis.

Differences from rtk_fdk:
  * GPU only: there is no CPU version (use rtk_fdk without a GPU). The backend counts
    as unavailable on a machine where ASTRA sees no CUDA device. `device="cpu"` is
    not honoured -- actual_device() says "gpu" -- since there is nothing to fall back to.
  * The projections and the volume both live in GPU memory at once; a volume too big
    for the card fails with an error saying so (use `size_xy` / `size_z`, or rtk_fdk).
  * `gpu_index` option: which GPU to use (default 0).

Options: `filter_name` ("hann" default, "ramp", "shepp-logan", "cosine", "hamming"),
`cut_frequency` (fraction of Nyquist where the window ends, default 1.0),
`size_xy` / `size_z` (output voxels per side / along the axis; default the detector
width / the number of rows), `gpu_index`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from reconstruction.backend import Geometry, ReconstructionBackend, register_backend
from reconstruction.backends._cone_common import cone_parameters, line_integrals, output_size

_ASTRA_FILTER = {"hann": "hann", "ramp": "ram-lak", "shepp-logan": "shepp-logan", "cosine": "cosine", "hamming": "hamming"}
_INSTALL_HINT = "Install with: pip install astra-toolbox (CUDA 12 wheels, Linux and Windows; needs an NVIDIA driver)"


def _astra():
    try:
        import astra
    except ImportError as e:
        raise ImportError(f"The 'astra_fdk' reconstruction backend needs the 'astra-toolbox' package. {_INSTALL_HINT}") from e
    return astra


@register_backend
class AstraFdkBackend(ReconstructionBackend):
    name = "astra_fdk"
    supported_geometries = {"cone"}
    requires = ("astra",)
    install_hint = _INSTALL_HINT
    filters = tuple(_ASTRA_FILTER)
    default_filter = "hann"

    def unavailable_reason(self) -> str:
        reason = super().unavailable_reason()
        if not reason and not self._cuda():
            return "needs an NVIDIA GPU with a working driver (ASTRA found no CUDA device); rtk_fdk reconstructs cone-beam data on the CPU"
        return reason

    def is_available(self) -> bool:
        return not self.unavailable_reason()

    @staticmethod
    def _cuda() -> bool:
        try:
            return bool(_astra().use_cuda())
        except Exception:  # noqa: BLE001 -- a broken driver is "no GPU", not a crash while listing backends
            return False

    def reconstruct(
        self,
        sinograms: np.ndarray,
        angles: np.ndarray,
        center,
        geometry: Geometry,
        device: Optional[str] = None,
        filter_name: str = "hann",
        cut_frequency: float = 1.0,
        size_xy: Optional[int] = None,
        size_z: Optional[int] = None,
        gpu_index: int = 0,
        **options,
    ) -> np.ndarray:
        if options:
            raise TypeError(f"astra_fdk got unexpected options {sorted(options)}")
        if filter_name not in _ASTRA_FILTER:
            raise ValueError(f"unknown filter {filter_name!r}; choose from {tuple(_ASTRA_FILTER)}")
        sdd, sod, pixel_mm, voxel_mm, axis_column = cone_parameters(geometry, center)
        astra = _astra()
        if not astra.use_cuda():
            raise RuntimeError(self.unavailable_reason() or "ASTRA found no CUDA device")

        data = np.asarray(sinograms)
        n_angles, n_rows, n_cols = data.shape
        n_xy, n_z = output_size(n_rows, n_cols, size_xy, size_z)

        vectors = np.zeros((n_angles, 12))
        for k, theta in enumerate(np.radians(np.asarray(angles, dtype=np.float64))):
            u = np.array([np.cos(theta), -np.sin(theta), 0.0])
            r = np.array([np.sin(theta), np.cos(theta), 0.0])
            vectors[k, 0:3] = -sod * r                                                              # source
            vectors[k, 3:6] = (sdd - sod) * r + ((n_cols - 1) / 2.0 - axis_column) * pixel_mm * u  # detector middle
            vectors[k, 6:9] = pixel_mm * u                                                          # one column step
            vectors[k, 9:12] = (0.0, 0.0, pixel_mm)                                                 # one row step

        half_xy, half_z = n_xy * voxel_mm / 2.0, n_z * voxel_mm / 2.0
        projection_geometry = astra.create_proj_geom("cone_vec", n_rows, n_cols, vectors)
        volume_geometry = astra.create_vol_geom(n_xy, n_xy, n_z, -half_xy, half_xy, -half_xy, half_xy, -half_z, half_z)

        ids = []
        try:
            # ASTRA wants (rows, angles, columns)
            ids.append(astra.data3d.create("-proj3d", projection_geometry, np.ascontiguousarray(line_integrals(data).transpose(1, 0, 2))))
            ids.append(astra.data3d.create("-vol", volume_geometry, 0.0))
            config = astra.astra_dict("FDK_CUDA")
            config["ProjectionDataId"], config["ReconstructionDataId"] = ids
            config["option"] = {"FilterType": _ASTRA_FILTER[filter_name], "FilterD": float(cut_frequency), "GPUindex": int(gpu_index)}
            algorithm = astra.algorithm.create(config)
            ids.append(algorithm)
            try:
                astra.algorithm.run(algorithm)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"ASTRA's FDK_CUDA failed ({e}); a {n_xy}x{n_xy}x{n_z} volume from {n_angles} projections of "
                    f"{n_rows}x{n_cols} may not fit in GPU memory -- try size_xy / size_z, or the rtk_fdk backend (CPU)"
                ) from e
            volume = astra.data3d.get(ids[1])
        finally:
            for i, handle in enumerate(ids):
                (astra.algorithm if i == 2 else astra.data3d).delete(handle)

        # attenuation in 1/mm, (slice, y, x) already in this project's orientation
        return np.ascontiguousarray(volume, dtype=np.float32) * np.float32(pixel_mm)   # 1/mm -> per detector pixel

    def actual_device(self, device: Optional[str] = None) -> str:
        return "gpu"
