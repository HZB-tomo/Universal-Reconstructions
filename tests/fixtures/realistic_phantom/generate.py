"""
Ties phantom.py and simulate.py together into one entry point, and
offers two ways to get the result into a form the rest of the package
understands.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tests.fixtures.realistic_phantom.phantom import make_phantom_volume
from tests.fixtures.realistic_phantom.simulate import (
    apply_cor_and_tilt,
    attenuation_scale_for_min_transmission,
    make_detector_patterns,
    project_parallel,
    synthesize_raw,
)


@dataclass
class PhantomProject:
    """
    One simulated run's output, entirely in memory. Carries ground
    truth (`volume`, `true_center_of_rotation`, `true_tilt_angle`)
    alongside the same shape of data (projections/flats/darks/angles) a
    real import would produce.
    """

    volume: np.ndarray  # ground-truth object, shape (height, width, width)
    projections: np.ndarray  # simulated raw detector readings, (n_angles, height, width)
    flat_field: np.ndarray  # simulated flat frames, (n_flats, height, width)
    dark_field: np.ndarray  # simulated dark frames, (n_darks, height, width)
    angles: np.ndarray  # degrees, shape (n_angles,)
    true_center_of_rotation: float  # pixels, deviation from image center at the middle row
    true_tilt_angle: float  # degrees of rotation-axis tilt (axis lean)
    voxel_size: float


def generate(
    shape: tuple[int, int, int] = (48, 64, 64),
    n_angles: int = 200,
    angle_span: float = 180.0,
    n_flats: int = 10,
    n_darks: int = 10,
    cor_deviation: float = 0.0,
    tilt_deg: float = 0.0,
    attenuation_scale: float | None = None,
    target_min_transmission: float = 0.14,
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
    Simulate one CT acquisition. Pure computation, no file I/O; see
    write_hdf5()/to_project_object() for the two ways to consume the
    result. Defaults produce a small, deterministic (noise_level=0.0)
    dataset for quick tests.

    See doc/REALISTIC_PHANTOM.md for what each parameter controls, and
    simulate.py's module docstring for the forward model.

    `angle_span` less than 180 produces a limited-angle scan (angles
    still start at 0, `n_angles` evenly spaced over `[0, angle_span)`).

    `attenuation_scale`, if given explicitly, is a total optical depth
    per unit of the ellipse table's density scale -- an arbitrary,
    ungrounded number; nothing about the ellipse table's values ties it
    to a realistic exposure. None (the default) instead solves for it
    via attenuation_scale_for_min_transmission(), so the single
    most-attenuating ray ends up at `target_min_transmission` of the
    free beam, the way a real acquisition's exposure/filtration is
    actually set up (real practice: at least ~10-15%, hence the default
    0.14). Passing attenuation_scale explicitly bypasses this, at
    whatever exposure that value happens to produce for this object.
    """
    rng = np.random.default_rng(seed)
    height, width = shape[0], shape[1]

    volume = make_phantom_volume(shape)
    angles = np.linspace(0, angle_span, n_angles, endpoint=False)
    line_integrals = project_parallel(volume, angles)
    line_integrals = apply_cor_and_tilt(line_integrals, cor_deviation, tilt_deg)

    if attenuation_scale is None:
        attenuation_scale = attenuation_scale_for_min_transmission(line_integrals, target_min_transmission)

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
    data_import/vendors/generic_parallel_synchrotron.py expects. The
    file can be imported through DataImporter.import_project() like a
    real vendor file.
    """
    import h5py

    with h5py.File(out_path, "w") as f:
        f.attrs["beamline"] = "generic_parallel_synchrotron"
        f.create_dataset("/entry/instrument/detector/data", data=phantom.projections)
        f.create_dataset("/entry/instrument/detector/flat_fields", data=phantom.flat_field)
        f.create_dataset("/entry/instrument/detector/dark_fields", data=phantom.dark_field)
        f.create_dataset("/entry/instrument/sample/rotation_angle", data=phantom.angles)
        f.create_dataset("/entry/instrument/detector/pixel_size", data=phantom.voxel_size)


def write_tiff(phantom: PhantomProject, out_dir) -> None:
    """
    Serialize a PhantomProject to a folder in the layout
    data_import/vendors/generic_parallel_synchrotron.py's TIFF plugin
    expects: projections/flats/darks subfolders of per-frame TIFF
    files, plus a metadata.json sidecar (angles, voxel_size). The
    folder can be imported through DataImporter.import_project() like
    a real vendor folder.
    """
    import json

    import tifffile
    from pathlib import Path

    out_dir = Path(out_dir)
    for subfolder, stack in [
        ("projections", phantom.projections),
        ("flats", phantom.flat_field),
        ("darks", phantom.dark_field),
    ]:
        folder = out_dir / subfolder
        folder.mkdir(parents=True, exist_ok=True)
        n_digits = max(5, len(str(stack.shape[0] - 1)))
        for i, frame in enumerate(stack):
            tifffile.imwrite(folder / f"frame_{i:0{n_digits}d}.tif", frame)

    (out_dir / "metadata.json").write_text(json.dumps({
        "angles": phantom.angles.tolist(),
        "voxel_size": phantom.voxel_size,
    }))


def to_project_object(phantom: PhantomProject):
    """
    Build a core.project.ProjectObject directly from a PhantomProject,
    entirely in memory. Calls known_fields.reconcile(), the same
    normalization DataImporter.import_project() applies.
    """
    from core.data_interface import DataInterface  # noqa: F401
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
