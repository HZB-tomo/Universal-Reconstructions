"""
Vendor plugin for BAMline's (BESSY II) on-the-fly continuous-rotation
HDF5 layout:

    /entry/data/data                                projections, 3D --
                                                      also where flats
                                                      live (see below)
    /entry/instrument/NDAttributes/RingCurrent       always present
    /entry/instrument/NDAttributes/Binning_X         always present
    /entry/instrument/NDAttributes/CT_Pixelsize      voxel size, scalar
                                                      (present sometimes,
                                                      not guaranteed)
    /entry/instrument/NDAttributes/DMM_Energy,       beam energy,
    /entry/instrument/NDAttributes/DCM_Energy        one of the two
                                                      depending on which
                                                      monochromator was
                                                      in the beam
    root: no vendor-name attribute exists at all -- see signature below.

BAMline is always parallel-beam.

Angles: no fixed path. Which rotation table was used for a given scan
determines which of several NDAttributes holds the real per-frame angle
-- everything else present is an unused stage sitting at its rest
position, reading pure encoder noise (~1e-4 deg) for the whole scan.
The real channel is identified by range (max - min), not by name
priority: an unused channel's range never exceeds the noise floor,
while the real one sweeps ~180 deg. See _find_angle_channel().

Flats/darks: BAMline's on-the-fly convention has no separate flat/dark
dataset at all. The scan starts with the rotation stage stationary (a
flat-field plateau, ~20 frames in practice but never a fixed count),
accelerates, sweeps past 180 deg (deliberately overshooting so the
motor has room to decelerate smoothly), then stops. Only the leading
plateau is used as flats here (see _find_projection_window()); the
trailing plateau and the deceleration overshoot are both discarded --
neither is a clean flat-field (object stayed in the beam) nor a wanted
projection (redundant angles past 180). There is no dark-field
measurement in the file at all: BAMline's dark current is a fixed
per-detector constant, not read from the scan -- supply it as
`ProjectConfig.dark_offset` (e.g. 100.0 for the usual detector), not
through this plugin. See preprocessing/normalization.py's
build_reference().

Matching test-data generator: tests/fixtures/generate_bamline.py.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from data_import.plugin import ImportPlugin, register_plugin
from data_import.path_map import PathMap
from data_import.signature import Hdf5Signature
from core.metadata import MetadataBundle

_PROJECTIONS = "/entry/data/data"
_RING_CURRENT = "/entry/instrument/NDAttributes/RingCurrent"
_BINNING_X = "/entry/instrument/NDAttributes/Binning_X"
_PIXEL_SIZE = "/entry/instrument/NDAttributes/CT_Pixelsize"
_ENERGY_PATHS = ("/entry/instrument/NDAttributes/DMM_Energy", "/entry/instrument/NDAttributes/DCM_Energy")

# Every angle-attribute name seen across BAMline's various rotation
# tables/eras. Only one is ever the real, moving one in a given file;
# see _find_angle_channel().
_ANGLE_CANDIDATES = (
    "/entry/instrument/NDAttributes/SAMPLE_MICOS_W1",
    "/entry/instrument/NDAttributes/SAMPLE_MICOS_W2",
    "/entry/instrument/NDAttributes/CT_MICOS_W",
    "/entry/instrument/NDAttributes/SAMPLE_HUBER_W",
    "/entry/instrument/NDAttributes/AEROTECH_W",
    "/entry/instrument/NDAttributes/SAMPLE_W",
    "/entry/instrument/NDAttributes/SAMPLE_MICOS_piezos",
)

# A stage that never rotated for this scan reads pure encoder noise,
# observed at roughly +/-1e-4 deg. A real sweep covers ~180 deg. 1.0 is
# nowhere near either number, so it can't misclassify one as the other.
_ACTIVE_CHANNEL_MIN_RANGE_DEG = 1.0

# How far a value has to move from the leading plateau's angle before
# it counts as "the motor has started turning", not noise on a
# stationary stage. Comfortably above the ~1e-4 deg noise floor.
_NOISE_EPS_DEG = 0.01

# Frames to skip right after the plateau ends, before treating data as
# a usable projection -- the motor is still accelerating and those
# frames risk motion blur. Not derived from data; a fixed margin.
_ACCEL_MARGIN_FRAMES = 3


def _find_angle_channel(handle) -> tuple[Optional[str], Optional[np.ndarray]]:
    """The candidate NDAttributes path with the largest value range,
    provided that range clears the noise floor -- see module docstring
    and _ACTIVE_CHANNEL_MIN_RANGE_DEG."""
    best_path, best_array, best_range = None, None, -1.0
    for path in _ANGLE_CANDIDATES:
        if path not in handle:
            continue
        array = np.asarray(handle[path][()], dtype=float)
        if array.size == 0:
            continue
        value_range = float(array.max() - array.min())
        if value_range > best_range:
            best_path, best_array, best_range = path, array, value_range
    if best_array is None or best_range < _ACTIVE_CHANNEL_MIN_RANGE_DEG:
        return None, None
    return best_path, best_array


def _find_projection_window(angles: np.ndarray) -> tuple[int, int, int]:
    """
    (rot_start, proj_start, proj_end): `rot_start` is the first frame
    that's genuinely moved off the leading plateau (frames before it
    are flats); `proj_start`/`proj_end` (both inclusive) bound the
    ~180 deg window used for reconstruction, starting a small
    acceleration margin after `rot_start`. See the module docstring.
    """
    n = len(angles)
    start_value = angles[0]

    rot_start = n
    for i in range(n):
        if abs(angles[i] - start_value) > _NOISE_EPS_DEG:
            rot_start = i
            break

    proj_start = min(rot_start + _ACCEL_MARGIN_FRAMES, n - 1)
    target = angles[proj_start] + 180.0
    proj_end = proj_start + int(np.argmin(np.abs(angles[proj_start:] - target)))
    return rot_start, proj_start, proj_end


@register_plugin
class BamlineHdf5Plugin(ImportPlugin):
    name = "bamline_hdf5"
    format = "hdf5"

    # No root attribute identifies BAMline by name at all -- these
    # files carry no such marker. Specificity instead comes from a
    # combination of always-present BAMline-specific NDAttributes
    # (RingCurrent, Binning_X are logged on every frame) plus a
    # structural requirement no unrelated vendor would coincidentally
    # satisfy: at least one candidate rotation-table attribute that
    # actually moved through a real sweep, not just noise.
    signature = Hdf5Signature(
        required_paths=[_PROJECTIONS, _RING_CURRENT, _BINNING_X],
        custom_predicate=lambda handle: (
            _find_angle_channel(handle)[0] is not None,
            "at least one rotation-table NDAttribute has a real (non-noise) angle sweep",
        ),
    )

    def get_path_map(self, handle) -> PathMap:
        angle_path, angles = _find_angle_channel(handle)
        if angle_path is None:
            raise ValueError(
                "bamline_hdf5: none of the known rotation-table attributes "
                f"({', '.join(_ANGLE_CANDIDATES)}) has a real angle sweep in this file"
            )
        rot_start, proj_start, proj_end = _find_projection_window(angles)

        return PathMap(
            projections=_PROJECTIONS,
            projections_range=slice(proj_start, proj_end + 1),
            flats=_PROJECTIONS,
            flats_range=slice(0, rot_start),
            angles=angle_path,
            extra={"angle_path": angle_path, "proj_start": proj_start, "proj_end": proj_end},
        )

    def parse_metadata(self, handle, path_map: PathMap) -> MetadataBundle:
        bundle = MetadataBundle()

        bundle.set("projections_path", path_map.projections, tier="mandatory_hard")
        bundle.set("flats_path", path_map.flats, tier="mandatory_hard")
        # No darks in the file at all -- see module docstring. Not even
        # declared missing: darks aren't a metadata field this vendor
        # can ever supply, unlike a field that's merely absent this time.

        full_angles = np.asarray(handle[path_map.angles][()], dtype=float)
        proj_start, proj_end = path_map.extra["proj_start"], path_map.extra["proj_end"]
        bundle.set("angles", full_angles[proj_start:proj_end + 1], tier="mandatory_soft", source="file")
        bundle.set("geometry_type", "parallel", tier="mandatory_soft", source="file")

        if _PIXEL_SIZE in handle:
            bundle.set("voxel_size", handle[_PIXEL_SIZE][()], tier="mandatory_soft", source="file")
        else:
            bundle.declare_missing("voxel_size", tier="mandatory_soft")

        energy_path = next((p for p in _ENERGY_PATHS if p in handle), None)
        if energy_path is not None:
            bundle.set("energy", handle[energy_path][()], tier="optional", source="file")

        bundle.declare_missing("center_of_rotation", tier="computed")
        bundle.declare_missing("tilt_angle", tier="computed")

        return bundle
