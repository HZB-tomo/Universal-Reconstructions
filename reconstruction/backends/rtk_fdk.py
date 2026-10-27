"""
Cone-beam FDK reconstruction backend, wrapping the Reconstruction Toolkit (RTK).

FDK (Feldkamp, Davis & Kress, 1984) is the standard analytic reconstruction
for a circular cone-beam scan: weight each projection by the cosine of its ray
angle, ramp-filter the rows, and backproject along the cone rays. It is exact
in the central plane and an approximation away from it (cone-beam artifacts
grow with the cone angle).

RTK (https://www.openrtk.org, Apache-2.0) provides the CPU implementation used
here, so it needs no GPU: `pip install itk-rtk` (the `cone` extra). RTK also
has CUDA versions; this backend uses the CPU one (multi-threaded by ITK).

Geometry (the cone-beam generalization of the parallel-beam convention):
the rotation axis is the vertical axis (detector rows run along it); at angle
theta the detector's column axis is u = (cos theta, -sin theta) in the
horizontal plane, the ray direction r = (sin theta, cos theta), the source sits
at -SOD * r and the detector plane at (SDD - SOD) * r. `center` is the detector
column onto which the rotation axis projects. Flat detector, circular orbit;
the result is (slices, y, x) with slices along the rotation axis, like
the parallel-beam backends. Voxels are cubic, with the size the detector pixel
has at the rotation axis (pixel / magnification).

Contract differences from the parallel-beam backends:
  * The input is the whole projection stack, (angles, detector rows, columns):
    every row is needed for every output slice, so a block of rows cannot be
    reconstructed on its own (no streaming, no single-slice preview).
  * `center` is one number (a tilted axis is not modelled).
  * geometry needs source_to_detector_distance, source_to_object_distance and
    detector_pixel_size (um).
  * Output is returned in "per pixel" units as the contract demands, where a
    pixel is the detector pixel; reconstruct() then converts to 1/cm.

Options: `filter_name` ("hann" default, "ramp", "shepp-logan", "cosine"),
`cut_frequency` (fraction of Nyquist where the window ends, default 1.0),
`size_xy` / `size_z` (output voxels per side / along the axis; default the
detector width / the number of rows).
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from reconstruction.backend import Geometry, ReconstructionBackend, register_backend

_FILTERS = ("hann", "ramp", "shepp-logan", "cosine")
_INSTALL_HINT = (
    "The 'rtk_fdk' reconstruction backend needs the 'itk-rtk' package. "
    "Install it with: pip install itk-rtk   (or: pip install -e \".[cone]\")"
)


def _rtk():
    try:
        with warnings.catch_warnings():
            # ITK's SWIG bindings warn about missing __module__ attributes on newer Pythons; nothing to act on.
            warnings.simplefilter("ignore", DeprecationWarning)
            import itk
            from itk import RTK as rtk
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e
    return itk, rtk


@register_backend
class RtkFdkBackend(ReconstructionBackend):
    name = "rtk_fdk"
    supported_geometries = {"cone"}
    requires = ("itk",)
    install_hint = "Install with: pip install itk-rtk (the 'cone' extra)"
    filters = _FILTERS
    default_filter = "hann"

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
        **options,
    ) -> np.ndarray:
        if options:
            raise TypeError(f"rtk_fdk got unexpected options {sorted(options)}")
        if filter_name not in _FILTERS:
            raise ValueError(f"unknown filter {filter_name!r}; choose from {_FILTERS}")
        sdd, sod, pixel_um = (
            geometry.source_to_detector_distance, geometry.source_to_object_distance, geometry.detector_pixel_size,
        )
        missing = [name for name, value in (
            ("source_to_detector_distance", sdd), ("source_to_object_distance", sod), ("detector_pixel_size", pixel_um),
        ) if not value]
        if missing:
            raise ValueError(f"cone-beam reconstruction needs {', '.join(missing)}")
        if not float(sdd) > float(sod) > 0:
            raise ValueError(f"need source_to_detector_distance ({sdd}) > source_to_object_distance ({sod}) > 0")

        centers = np.atleast_1d(np.asarray(center, dtype=np.float64))
        if np.ptp(centers) > 1e-6:
            raise ValueError("cone-beam reconstruction takes one center of rotation; a per-slice (tilt) correction is not supported")

        itk, rtk = _rtk()
        data = np.asarray(sinograms)
        n_angles, n_rows, n_cols = data.shape
        # attenuation: -log of transmission (values <= 0 count as 1, like the other backends)
        transmission = np.array(data, dtype=np.float32)
        transmission[transmission <= 0.0] = 1.0
        line_integrals = np.ascontiguousarray(-np.log(transmission), dtype=np.float32)

        pixel_mm = float(pixel_um) / 1000.0
        magnification = float(sdd) / float(sod)
        voxel_mm = pixel_mm / magnification
        n_xy = int(size_xy or n_cols)
        n_z = int(size_z or n_rows)
        axis_column = float(centers[0])

        ImageType = itk.Image[itk.F, 3]
        projections = itk.image_from_array(line_integrals)           # numpy (angle, row, column) = ITK (x=column, y=row, z=angle)
        projections.SetSpacing([pixel_mm, pixel_mm, 1.0])
        projections.SetOrigin([-(n_cols - 1) / 2.0 * pixel_mm, -(n_rows - 1) / 2.0 * pixel_mm, 0.0])

        # RTK convention (found against analytic phantoms, see tests/test_cone_beam.py): its angle runs the
        # opposite way to ours, and the detector offset is the opposite of the axis position we give.
        rtk_geometry = rtk.ThreeDCircularProjectionGeometry.New()
        offset_x = -(axis_column - (n_cols - 1) / 2.0) * pixel_mm
        for angle in np.asarray(angles, dtype=np.float64):
            rtk_geometry.AddProjection(float(sod), float(sdd), -float(angle), offset_x, 0.0)

        # RTK volume axes: x horizontal, y VERTICAL (along the rotation axis), z horizontal.
        source = rtk.ConstantImageSource[ImageType].New()
        source.SetSize([n_xy, n_z, n_xy])
        source.SetSpacing([voxel_mm] * 3)
        source.SetOrigin([-(n_xy - 1) / 2.0 * voxel_mm, -(n_z - 1) / 2.0 * voxel_mm, -(n_xy - 1) / 2.0 * voxel_mm])
        source.SetConstant(0.0)

        fdk = rtk.FDKConeBeamReconstructionFilter[ImageType].New()
        fdk.SetInput(0, source.GetOutput())
        fdk.SetInput(1, projections)
        fdk.SetGeometry(rtk_geometry)
        ramp = fdk.GetRampFilter()
        if filter_name == "hann":
            ramp.SetHannCutFrequency(float(cut_frequency))
        elif filter_name == "shepp-logan":
            ramp.SetSheppLoganCutFrequency(float(cut_frequency))
        elif filter_name == "cosine":
            ramp.SetCosineCutFrequency(float(cut_frequency))
        fdk.Update()

        volume = itk.array_from_image(fdk.GetOutput())               # (RTK z, RTK y = along the axis, RTK x)
        # ours: (slice along the axis, y, x) with RTK z = -our y
        ours = np.transpose(volume, (1, 0, 2))[:, ::-1, :]
        return np.ascontiguousarray(ours, dtype=np.float32) * np.float32(pixel_mm)   # 1/mm -> per detector pixel

    def actual_device(self, device: Optional[str] = None) -> str:
        return "cpu"
