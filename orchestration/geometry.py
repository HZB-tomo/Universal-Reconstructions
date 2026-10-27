"""
geometry_from_project(): the reconstruction Geometry of an imported project,
built the same way by whole runs, streamed runs and previews.

For parallel beam the detector pixel size is the pixel size. For cone beam
there are two sizes: the detector's own pixel pitch, and the voxel size at the
rotation axis, which is smaller by the magnification M = SDD / SOD. The GUI's
"Pixel size" and the `voxel_size` field mean the latter (the size of a pixel as
it appears at the sample), so for a cone-beam project the detector pitch is
voxel_size * M -- unless the detector pitch itself was given (by the file or the
user), which then wins.
"""
from __future__ import annotations

from core.data_interface import DataInterface
from core.project import ProjectObject
from reconstruction import Geometry


def geometry_from_project(project: ProjectObject) -> Geometry:
    di = DataInterface(project)
    geometry_type = di.get_metadata("geometry_type")
    sdd = di.get_metadata("source_to_detector_distance")
    sod = di.get_metadata("source_to_object_distance")
    detector_pixel = di.get_metadata("detector_pixel_size")

    if geometry_type == "cone" and sdd and sod:
        pitch_is_given = di.get_metadata_source("detector_pixel_size") in ("file", "user")
        voxel = di.get_metadata("voxel_size")
        voxel_is_known = di.get_metadata_source("voxel_size") not in (None, "default")
        if not pitch_is_given and voxel is not None and voxel_is_known:
            detector_pixel = float(voxel) * float(sdd) / float(sod)

    return Geometry(
        geometry_type=geometry_type,
        source_to_detector_distance=sdd,
        source_to_object_distance=sod,
        detector_pixel_size=detector_pixel,
    )
