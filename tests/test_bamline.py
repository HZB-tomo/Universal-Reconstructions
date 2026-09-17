import numpy as np
import pytest

from core.data_interface import DataInterface
from data_import.importer import DataImporter
from orchestration import PipelineRun, ProjectConfig
from tests.fixtures.generate_bamline import generate


@pytest.mark.parametrize(
    "angle_attribute",
    [
        "/entry/instrument/NDAttributes/SAMPLE_MICOS_W2",  # matches the real file this plugin was built from
        "/entry/instrument/NDAttributes/AEROTECH_W",  # a different rotation table -- same detection must still work
    ],
)
def test_detects_bamline_regardless_of_which_rotation_table_was_used(tmp_path, angle_attribute):
    path = tmp_path / "scan.h5"
    generate(path, angle_attribute=angle_attribute)

    plugin = DataImporter().detect(path, format="hdf5")

    assert plugin.name == "bamline_hdf5"


def test_projection_window_excludes_plateaus_and_is_close_to_180_degrees(tmp_path):
    n_flats, n_rotate, overshoot_deg = 15, 200, 5.0
    path = tmp_path / "scan.h5"
    volume, n_flats, n_rotate, overshoot_deg = generate(
        path, n_flats=n_flats, n_rotate=n_rotate, overshoot_deg=overshoot_deg
    )
    step_deg = (180.0 + overshoot_deg) / n_rotate

    plugin = DataImporter().detect(path, format="hdf5")
    import h5py

    with h5py.File(path, "r") as handle:
        path_map = plugin.get_path_map(handle)

    # The leading plateau is exactly n_flats frames of dead-flat 0.0 --
    # rot_start should land exactly there, not fuzzily nearby.
    assert path_map.flats_range == slice(0, n_flats)

    n_projections = path_map.projections_range.stop - path_map.projections_range.start
    assert n_projections > 0
    # Well short of the full rotate+trailing region: the deceleration
    # overshoot past 180 deg must have been excluded, not included.
    assert path_map.projections_range.stop < n_flats + n_rotate

    with h5py.File(path, "r") as handle:
        bundle = plugin.parse_metadata(handle, path_map)

    angles = bundle.get("angles")
    angle_span = float(angles.max() - angles.min())
    # Can't land on exactly 180.0 given discrete sampling; must land
    # within one frame's worth of it, not just "somewhere past 180".
    assert abs(angle_span - 180.0) < step_deg * 1.5


def test_flats_are_only_the_leading_plateau_not_the_trailing_one(tmp_path):
    path = tmp_path / "scan.h5"
    volume, n_flats, n_rotate, overshoot_deg = generate(path, n_flats=12, n_trailing=8)

    plugin = DataImporter().detect(path, format="hdf5")
    import h5py

    with h5py.File(path, "r") as handle:
        path_map = plugin.get_path_map(handle)

    assert path_map.flats_range.stop - path_map.flats_range.start == 12


def test_full_import_resolves_with_no_missing_soft_fields(tmp_path):
    path = tmp_path / "scan.h5"
    generate(path)

    config = ProjectConfig(project_id="bamline_test", path=str(path), format="hdf5", storage_mode="ram")
    run = PipelineRun(config)
    run.import_data()
    run.complete()  # derives angle_span/detector_pixel_size etc. from what the plugin read

    assert run.project.missing_soft_fields() == []
    assert run.project.metadata.get("geometry_type") == "parallel"
    # CT_Pixelsize/DMM_Energy are per-frame NDAttributes in real BAMline
    # files (see generate_bamline.py), not scalars -- voxel_size must
    # come out as one number, not the whole per-frame array.
    assert np.ndim(run.project.metadata.get("voxel_size")) == 0
    assert run.project.metadata.get("voxel_size") == pytest.approx(0.65)

    di = DataInterface(run.project)
    n_proj, height, width = di.shape_of("projections")
    assert n_proj == len(run.project.metadata.get("angles"))

    flats = run.project.data_handles["flats"].get()
    assert flats.shape[0] == 15  # generate()'s n_flats default
    assert "darks" not in run.project.data_handles


def test_an_unused_rotation_stage_reading_only_noise_does_not_get_picked(tmp_path):
    """Sanity check on the fixture itself as much as the plugin: every
    candidate NOT designated as the real channel must actually look
    like noise (small range), or this test proves nothing."""
    path = tmp_path / "scan.h5"
    generate(path, angle_attribute="/entry/instrument/NDAttributes/SAMPLE_MICOS_W2")

    import h5py

    with h5py.File(path, "r") as handle:
        unused = np.asarray(handle["/entry/instrument/NDAttributes/AEROTECH_W"][()])

    assert (unused.max() - unused.min()) < 1.0  # comfortably under bamline.py's _ACTIVE_CHANNEL_MIN_RANGE_DEG
