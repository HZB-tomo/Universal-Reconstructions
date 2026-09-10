"""DESY raw TIFF reader with parameters.mat metadata.

Only paths and small metadata are read here; the TIFF backend handles pixels.
The supplied MAT is a reconstruction parameter file, so it may omit actual
acquisition measurements. Saved processing settings are not applied to raw data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from core.metadata import MetadataBundle
from data_import.path_map import PathMap
from data_import.plugin import ImportPlugin, register_plugin
from data_import.signature import SignatureResult, TiffSignature

# Replace these with globs RELATIVE to the scan folder, not absolute paths.
_PROJECTIONS = "Proj/proj_*.tif*"
_FLATS = "Flats/flat_*.tif*"
_DARKS = "Darks/dark_*.tif*"
_METADATA = "parameters.mat"
_LENGTH_TO_MM = {"m": 1000.0, "mm": 1.0, "um": 0.001} # I guess the unit is also important for other 
#stuff than pixels


def _present(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return np.size(value) > 0


def _record(bundle, key, value):
    """Keep useful vendor settings, including zeros, without MATLAB objects."""
    if _present(value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, np.generic):
            value = value.item()
        bundle.set(key, value, tier="optional")
    else:
        bundle.declare_missing(key, tier="optional")


def _positive_scalar(value, key):
    array = np.asarray(value, dtype=float)
    if array.size != 1 or not np.isfinite(array).all() or array.item() <= 0:
        raise ValueError(f"DESY {key} must be a finite positive scalar.")
    return float(array.item())


class DesyTiffSignature(TiffSignature):
    def check(self, folder):
        folder = Path(folder)
        result = super().check(folder)
        # Base TIFF signatures search recursively. Match the backend's root.
        for pattern in self.required_sidecar_patterns:
            if not any(p.is_file() for p in folder.glob(pattern)):
                result.failed.append(f"missing file relative to scan root: {pattern}")
        return SignatureResult(not result.failed, result.checked, result.failed)


@register_plugin
class DesyTiffPlugin(ImportPlugin):
    name = "desy_tiff"
    format = "tiff"
    signature = DesyTiffSignature(
        required_sidecar_patterns=[_PROJECTIONS, _FLATS, _DARKS, _METADATA],
        min_file_count=1,
    )

    def __init__(self, *, angle_unit=None, pixel_size_unit=None,
                 energy_unit=None, distance_unit=None):
        """Optional, confirmed conventions for fields in parameters.mat.

        angle_unit: 'degree'/'radian'; pixel_size_unit and distance_unit:
        'm'/'mm'/'um'; energy_unit: 'eV'/'keV'. None leaves the canonical
        measurement missing. eff_pixel_size must describe the raw TIFFs
        before enabling pixel_size_unit; raw_bin is never applied here.
        """
        for key, unit, allowed in (
            ("angle_unit", angle_unit, ("degree", "radian")),
            ("pixel_size_unit", pixel_size_unit, _LENGTH_TO_MM),
            ("energy_unit", energy_unit, ("eV", "keV")),
            ("distance_unit", distance_unit, _LENGTH_TO_MM),
        ):
            if unit is not None and unit not in allowed:
                raise ValueError(f"Unsupported DESY {key}: {unit!r}")
            setattr(self, key, unit)

    def get_path_map(self, handle) -> PathMap:
        return PathMap(
            projections=_PROJECTIONS, flats=_FLATS, darks=_DARKS,
            metadata_source=_METADATA,
        )

    def parse_metadata(self, handle, path_map: PathMap) -> MetadataBundle:
        folder = Path(handle)
        bundle = MetadataBundle()
        counts = {}
        for key, count_key in (("projections", "projection_count"),
                               ("flats", "flat_count"), ("darks", "dark_count")):
            pattern = getattr(path_map, key)
            count = sum(p.is_file() for p in folder.glob(pattern)) if pattern else 0
            counts[key] = count
            if count:
                bundle.set(f"{key}_path", pattern, tier="mandatory_hard")
            else:
                bundle.declare_missing(f"{key}_path", tier="mandatory_hard")
            bundle.set(count_key, count, tier="optional", source="computed")

        # Ignore unrelated saved GUI state and MATLAB function handles.
        mat = loadmat(folder / path_map.metadata_source, simplify_cells=True,
                      variable_names=["par", "tomo", "phase_retrieval", "ring_filter",
                                      "angles", "scan_name", "DATA_SET_NUM"])
        par = mat.get("par")
        if not isinstance(par, dict):
            raise ValueError("DESY parameters.mat must contain a 'par' struct.")

        # The agreed DESY/P07 convention is parallel beam; this is a convention
        # default, not a geometry flag measured or found in the supplied MAT.
        bundle.set("geometry_type", "parallel", tier="mandatory_soft", source="default")
        for key in ("angles", "angle_span", "voxel_size", "detector_pixel_size"):
            bundle.declare_missing(key, tier="mandatory_soft")

        angles = mat.get("angles")
        if _present(angles):
            angles = np.asarray(angles, dtype=float)
            if angles.ndim > 1 or not np.isfinite(angles).all():
                raise ValueError("DESY angles must be a finite per-projection vector.")
            angles = angles.reshape(-1)
            if counts["projections"] and len(angles) != counts["projections"]:
                raise ValueError(
                    f"DESY angle-count mismatch: {len(angles)} angles for "
                    f"{counts['projections']} projections."
                )
            _record(bundle, "desy_angles", angles)
            if self.angle_unit is not None:
                degrees = np.rad2deg(angles) if self.angle_unit == "radian" else angles
                bundle.set("angles", degrees, tier="mandatory_soft")

        for raw_key, key, unit, scale, tier in (
            ("eff_pixel_size", "detector_pixel_size", self.pixel_size_unit,
             _LENGTH_TO_MM.get(self.pixel_size_unit, 1) * 1000, "mandatory_soft"),
            ("energy", "energy", self.energy_unit,
             0.001 if self.energy_unit == "eV" else 1, "optional"),
            ("sample_detector_distance", "sample_detector_distance", self.distance_unit,
             _LENGTH_TO_MM.get(self.distance_unit, 1), "optional"),
        ):
            value = par.get(raw_key)
            _record(bundle, f"desy_{raw_key}", value)
            if unit is not None and _present(value):
                bundle.set(key, _positive_scalar(value, raw_key) * scale, tier=tier)
        for key in ("angle_unit", "pixel_size_unit", "energy_unit", "distance_unit"):
            bundle.set(f"desy_{key}", getattr(self, key), tier="optional", source="user")

        for key, raw_key in (("raw_binning", "raw_bin"),
                             ("raw_image_shape", "im_shape_raw"), ("image_dtype", "dtype")):
            _record(bundle, key, par.get(raw_key))

        # These describe the earlier reconstruction, potentially after binning.
        # In particular, its axis offset is NOT the raw-image calibration.
        for group, fields in {
            "tomo": ("rot_axis_offset", "rot_axis_position", "rot_axis_tilt_camera",
                     "rot_axis_tilt_lamino", "reco_mode", "algorithm", "iterations",
                     "fbp_filter_type", "take_neg_log", "vol_shape", "vol_size"),
            "phase_retrieval": ("apply", "method"),
            "ring_filter": ("apply", "method"),
        }.items():
            values = mat.get(group, {})
            if isinstance(values, dict):
                for key in fields:
                    _record(bundle, f"desy_{group}_{key}", values.get(key))
        _record(bundle, "dataset_number", mat.get("DATA_SET_NUM"))
        _record(bundle, "scan_name", mat.get("scan_name"))
        bundle.declare_missing("center_of_rotation", tier="computed")
        bundle.declare_missing("tilt_angle", tier="computed")
        return bundle
