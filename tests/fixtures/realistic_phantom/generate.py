"""
Ties phantom.py and simulate.py together into one entry point, and
offers two separate ways to get the result into something the rest of
the package understands -- see the module docstring in __init__.py and
doc/REALISTIC_PHANTOM.md for why these are kept as three distinct
functions instead of one that does everything.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tests.fixtures.realistic_phantom.phantom import make_phantom_volume
from tests.fixtures.realistic_phantom.simulate import (
    apply_cor_and_tilt,
    make_detector_patterns,
    project_parallel,
    synthesize_raw,
)


@dataclass
class PhantomProject:
    """
    Everything one simulated run produced -- purely in memory, no file
    involved. Carries the ground truth (`volume`, `true_center_of_rotation`,
    `true_tilt_angle`) that a real import could never have, alongside the
    same shape of data (projections/flats/darks/angles) a real one would
    produce -- exactly what makes this useful for checking a later
    calibration or reconstruction result against a known answer.
    """

    volume: np.ndarray  # ground-truth object, shape (height, width, width)
    projections: np.ndarray  # simulated raw detector readings, (n_angles, height, width)
    flat_field: np.ndarray  # simulated flat frames, (n_flats, height, width)
    dark_field: np.ndarray  # simulated dark frames, (n_darks, height, width)
    angles: np.ndarray  # degrees, shape (n_angles,)
    true_center_of_rotation: float  # pixels, deviation from image center at the middle row
    true_tilt_angle: float  # degrees of rotation-axis tilt (axis lean) -- see simulate.py's
    # apply_cor_and_tilt() docstring; NOT an in-plane "detector roll" angle
    voxel_size: float


def generate(
    shape: tuple[int, int, int] = (48, 64, 64),
    n_angles: int = 200,
    n_flats: int = 10,
    n_darks: int = 10,
    cor_deviation: float = 0.0,
    tilt_deg: float = 0.0,
    attenuation_scale: float = 2.0,
    flat_intensity: float = 1.0,
    dark_level: float = 0.01,
    inhomogeneity_strength: float = 0.05,
    n_scratches: int = 3,
    scratch_strength: float = 0.15,
    n_defective_pixels: int = 5,
    noise_level: float = 0.0,
    voxel_size: float = 1.0,
    seed: int | None = None,
) -> PhantomProject:
    """
    Simulate one realistic CT acquisition. Pure computation, no file
    I/O -- see write_hdf5()/to_project_object() for the two ways to get
    this into something the rest of the package can use. All defaults
    produce a small, fast, deterministic (noise_level=0.0) dataset
    suitable for quick tests; increase `shape`/`n_angles` for something
    closer to a real acquisition.

    See doc/REALISTIC_PHANTOM.md for what every parameter controls
    physically and simulate.py's module docstring for the forward model.
    """
    rng = np.random.default_rng(seed)

    volume = make_phantom_volume(shape)
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    line_integrals = project_parallel(volume, angles)
    line_integrals = apply_cor_and_tilt(line_integrals, cor_deviation, tilt_deg)

    height, width = shape[0], shape[1]
    gain, dark_offset = make_detector_patterns(
        (height, width), inhomogeneity_strength, n_scratches, scratch_strength, n_defective_pixels, dark_level, rng
    )
    projections, flat_field, dark_field = synthesize_raw(
        line_integrals, gain, dark_offset, flat_intensity, attenuation_scale, n_flats, n_darks, noise_level, rng
    )

    return PhantomProject(
        volume=volume,
        projections=projections,
        flat_field=flat_field,
        dark_field=dark_field,
        angles=angles,
        true_center_of_rotation=cor_deviation,
        true_tilt_angle=tilt_deg,
        voxel_size=voxel_size,
    )


def write_hdf5(phantom: PhantomProject, out_path) -> None:
    """
    Serialize a PhantomProject to disk in the layout
    data_import/vendors/generic_parallel_synchrotron.py's plugin
    expects -- the file this writes can be imported through
    DataImporter.import_project() exactly like a real vendor file,
    exercising the actual plugin-detection/completion pipeline rather
    than just the in-memory shortcut below.
    """
    import h5py  # imported lazily, same pattern as everywhere else in this codebase

    with h5py.File(out_path, "w") as f:
        f.attrs["beamline"] = "generic_parallel_synchrotron"
        f.create_dataset("/entry/instrument/detector/data", data=phantom.projections)
        f.create_dataset("/entry/instrument/detector/flat_fields", data=phantom.flat_field)
        f.create_dataset("/entry/instrument/detector/dark_fields", data=phantom.dark_field)
        f.create_dataset("/entry/instrument/sample/rotation_angle", data=phantom.angles)
        f.create_dataset("/entry/instrument/detector/pixel_size", data=phantom.voxel_size)


def to_project_object(phantom: PhantomProject):
    """
    Build a core.project.ProjectObject directly from a PhantomProject,
    entirely in memory -- no file written or read. The fast path for
    testing completion/preprocessing/calibration without paying for an
    hdf5 round-trip every time; write_hdf5() + DataImporter is the slow,
    fully-realistic path that also exercises plugin detection.

    Mirrors exactly what a real import of a generic_parallel_synchrotron
    file would produce, including calling known_fields.reconcile() --
    the same normalization DataImporter.import_project() applies.
    """
    from core.data_interface import DataInterface  # noqa: F401  (re-exported for convenience at the call site)
    from core.handles import RamHandle
    from core.known_fields import reconcile
    from core.metadata import MetadataBundle
    from core.project import ProjectObject

    meta = MetadataBundle()
    meta.set("projections_path", "<in-memory>", tier="mandatory_hard")
    meta.set("flats_path", "<in-memory>", tier="mandatory_hard")
    meta.set("darks_path", "<in-memory>", tier="mandatory_hard")
    meta.set("angles", phantom.angles, tier="mandatory_soft", source="file")
    meta.set("voxel_size", phantom.voxel_size, tier="mandatory_soft", source="file")
    meta.set("geometry_type", "parallel", tier="mandatory_soft", source="file")
    reconcile(meta)

    return ProjectObject(
        format="hdf5",
        storage_mode="ram",
        metadata=meta,
        data_handles={
            "projections": RamHandle(phantom.projections),
            "flats": RamHandle(phantom.flat_field),
            "darks": RamHandle(phantom.dark_field),
        },
        vendor_plugin_name="generic_parallel_synchrotron_hdf5",
    )
