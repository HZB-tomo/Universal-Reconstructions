"""
Tests for the Qt-free parts of "Reconstruct all slices": gui/reconstruct_plan.py,
gui/volume_view.py, and ram_check.estimate_inmemory_run_bytes().
"""
import h5py
import numpy as np
import pytest

from gui import reconstruct_plan
from gui.reconstruct_plan import RememberedPaths, can_stream_geometry, describe_error, make_plan, streaming_config
from gui.slicing import axis_labels, section_index
from gui.volume_view import open_volume
from orchestration import PipelineRun, ProjectConfig, ram_check


class _FakeVirtualMemory:
    def __init__(self, available):
        self.available = available


def _set_available(monkeypatch, available):
    monkeypatch.setattr(ram_check.psutil, "virtual_memory", lambda: _FakeVirtualMemory(available))


def test_inmemory_estimate_includes_the_sinogram_stack():
    n_angles, n_rows, width = 100, 10, 20
    sinograms = n_angles * n_rows * width * 4
    assert ram_check.estimate_inmemory_run_bytes(n_angles, n_rows, width) == sinograms + ram_check.estimate_volume_bytes(n_rows, width)


def test_plan_fits_when_ram_is_plentiful(monkeypatch):
    _set_available(monkeypatch, 100 * 1024**3)
    plan = make_plan(n_angles=100, n_rows=50, width=100, geometry_type="parallel")
    assert plan.fits_in_ram and plan.can_stream and plan.stream_blocker == ""


def test_plan_does_not_fit_when_ram_is_tiny(monkeypatch):
    _set_available(monkeypatch, 1024)
    plan = make_plan(n_angles=100, n_rows=50, width=100, geometry_type="parallel")
    assert not plan.fits_in_ram
    assert plan.available_bytes == 1024
    assert plan.needed_bytes > 1024


def test_needed_bytes_includes_the_safety_margin(monkeypatch):
    _set_available(monkeypatch, 1024)
    raw = ram_check.estimate_inmemory_run_bytes(100, 50, 100)
    assert make_plan(100, 50, 100, "parallel").needed_bytes == int(raw * ram_check.SAFETY_FACTOR)


def test_streaming_only_for_parallel_geometry(monkeypatch):
    _set_available(monkeypatch, 1024)
    assert can_stream_geometry("parallel")
    assert not can_stream_geometry("cone")
    assert not can_stream_geometry(None)
    cone = make_plan(10, 10, 10, "cone")
    assert not cone.can_stream and "cone" in cone.stream_blocker
    unset = make_plan(10, 10, 10, None)
    assert not unset.can_stream and "isn't set" in unset.stream_blocker


def test_streaming_config_forces_full_hdf5_stream_without_touching_the_original():
    config = ProjectConfig(path="a.h5", rows=slice(5, 9), crop=((0, 4), (0, 2), (0, 2)), export_format="tiff")
    streamed, note = streaming_config(config, "out/vol.h5")
    assert streamed.stream and streamed.rows == slice(5, 9) and streamed.crop is None  # slice range kept, crop not applied
    assert streamed.export_format == "hdf5" and streamed.export_path == "out/vol.h5"
    assert note == ""
    assert not config.stream and config.rows == slice(5, 9) and config.export_format == "tiff"


def test_streaming_config_falls_back_to_float32_when_uint16_has_no_clip():
    config = ProjectConfig(path="a.h5", dtype="uint16")
    streamed, note = streaming_config(config, "vol.h5")
    assert streamed.dtype == "float32" and "float32" in note
    clipped, note = streaming_config(ProjectConfig(path="a.h5", dtype="uint16", dtype_clip_min=0.0, dtype_clip_max=1.0), "vol.h5")
    assert clipped.dtype == "uint16" and note == ""


def test_describe_error_names_memory_errors_even_when_the_message_is_empty():
    assert "out of memory" in describe_error(MemoryError())
    assert "HDF5" in describe_error(MemoryError())
    assert describe_error(ValueError("bad")) == "bad"
    assert describe_error(RuntimeError()) == "RuntimeError"


def test_remembered_paths_default_next_to_source_then_follows_last_choice(tmp_path):
    paths = RememberedPaths()
    config = ProjectConfig(project_id="scan", path=str(tmp_path / "data" / "scan.h5"))
    assert paths.default_volume_path(config) == str(tmp_path / "data" / "scan_volume.h5")

    paths.remember_volume_path(str(tmp_path / "out" / "chosen.h5"))
    assert paths.default_volume_path(config) == str(tmp_path / "out" / "scan_volume.h5")


def test_open_volume_frames_reads_one_slice_at_a_time(tmp_path):
    path = tmp_path / "vol.h5"
    volume = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    with h5py.File(path, "w") as f:
        f.create_dataset("/volume", data=volume)

    get_section, shape = open_volume(str(path), "/volume")
    assert shape == (3, 4, 5)
    np.testing.assert_array_equal(get_section(2), volume[2])
    np.testing.assert_array_equal(get_section(section_index(1, 3)), volume[:, 3, :])
    np.testing.assert_array_equal(get_section(section_index(2, 4)), volume[:, :, 4])
    path.unlink()  # file is not held open between reads


def test_gui_style_streaming_reuses_an_already_imported_project(tmp_path):
    """What AdjustmentPanel._reconstruct_streaming() does: a second
    PipelineRun, sharing the first one's imported project, streams the
    whole volume into the chosen HDF5 file."""
    pytest.importorskip("algotom")
    from tests.fixtures.realistic_phantom import generate as generate_phantom
    from tests.fixtures.realistic_phantom import write_hdf5

    phantom = generate_phantom(
        shape=(8, 32, 32), n_angles=60, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=1,
    )
    source = tmp_path / "scan.h5"
    write_hdf5(phantom, source)

    base = ProjectConfig(path=str(source), format="hdf5", storage_mode="disk", center=16.0, tilt_deg=0.0)
    run = PipelineRun(base)
    run.import_data()
    run.complete()

    config, _ = streaming_config(base, str(tmp_path / "out" / "vol.h5"))
    streamer = PipelineRun(config)
    streamer.project = run.project
    written = streamer.stream()

    with h5py.File(written, "r") as f:
        assert f["/volume"].shape == (8, 32, 32)
        assert np.isfinite(f["/volume"][:]).all()


def test_streaming_reports_progress_per_chunk_with_device(tmp_path):
    pytest.importorskip("algotom")
    from tests.fixtures.realistic_phantom import generate as generate_phantom
    from tests.fixtures.realistic_phantom import write_hdf5

    source = tmp_path / "scan.h5"
    write_hdf5(generate_phantom(
        shape=(8, 32, 32), n_angles=60, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=1,
    ), source)
    events = []
    config = ProjectConfig(
        path=str(source), format="hdf5", storage_mode="disk", center=16.0, tilt_deg=0.0,
        stream=True, stream_chunk_rows=4, export_path=str(tmp_path / "vol.h5"),
    )
    assert PipelineRun(config, progress=events.append).run_all().success

    stages = [e.stage for e in events]
    assert stages[0] == "Importing" and "Reconstructing" in stages and "Writing" in stages
    streamed = [e for e in events if e.total == 8]
    assert {e.device for e in streamed} == {"cpu"} or {e.device for e in streamed} == {"gpu"}
    assert max(e.done for e in streamed) == 8
    assert [e.done for e in streamed if e.stage == "Reconstructing"] == [0, 4]


def test_eager_run_reports_every_stage_in_order(tmp_path):
    pytest.importorskip("algotom")
    from tests.fixtures.realistic_phantom import generate as generate_phantom
    from tests.fixtures.realistic_phantom import write_hdf5

    source = tmp_path / "scan.h5"
    write_hdf5(generate_phantom(
        shape=(4, 32, 32), n_angles=60, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=1,
    ), source)
    events = []
    config = ProjectConfig(path=str(source), format="hdf5", center=16.0, tilt_deg=0.0, export_path=str(tmp_path / "v.h5"))
    assert PipelineRun(config, progress=events.append).run_all().success
    seen = list(dict.fromkeys(e.stage for e in events))
    assert seen == ["Importing", "Completing metadata", "Normalizing projections", "Calibrating",
                    "Reconstructing", "Postprocessing", "Writing volume"]


def test_eager_run_with_progress_reports_slices_and_matches_the_unchunked_result(tmp_path, monkeypatch):
    pytest.importorskip("algotom")
    from tests.fixtures.realistic_phantom import generate as generate_phantom
    from tests.fixtures.realistic_phantom import write_hdf5

    source = tmp_path / "scan.h5"
    write_hdf5(generate_phantom(
        shape=(8, 32, 32), n_angles=60, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=1,
    ), source)
    monkeypatch.setattr(ram_check, "estimate_chunk_rows", lambda width, max_rows, *rest: 2)

    def run(progress, name):
        config = ProjectConfig(path=str(source), format="hdf5", center=16.0, tilt_deg=0.0, export_path=str(tmp_path / name))
        r = PipelineRun(config, progress=progress)
        assert r.run_all().success
        with h5py.File(tmp_path / name, "r") as f:
            return f["/volume"][:]

    events = []
    chunked = run(events.append, "a.h5")
    plain = run(None, "b.h5")
    plain_repeat = run(None, "c.h5")  # algotom's FBP isn't bit-deterministic; see tests/test_streaming.py
    assert [e.done for e in events if e.stage == "Reconstructing"] == [0, 2, 4, 6, 8]
    assert chunked.shape == plain.shape == (8, 32, 32)
    np.testing.assert_array_equal(plain, plain_repeat)  # CPU FBP is exactly reproducible now
    np.testing.assert_allclose(chunked, plain, rtol=1e-5, atol=1e-6)  # chunking changes nothing


def test_unique_path_never_overwrites(tmp_path):
    first = tmp_path / "vol.h5"
    assert reconstruct_plan.unique_path(first) == str(first)
    first.write_bytes(b"x")
    assert reconstruct_plan.unique_path(first) == str(tmp_path / "vol_1.h5")
    (tmp_path / "vol_1.h5").write_bytes(b"x")
    assert reconstruct_plan.unique_path(first) == str(tmp_path / "vol_2.h5")


def test_save_ram_volume_writes_data_settings_and_reports_progress(tmp_path):
    volume = np.random.default_rng(0).random((5, 4, 6)).astype(np.float32)
    events = []
    path = reconstruct_plan.save_ram_volume(
        volume, str(tmp_path / "v.h5"), {"center": np.float64(3.5)}, progress=events.append, project_id="p", chunk_slices=2,
    )
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["/volume"][:], volume)
        assert '"center": 3.5' in f["/volume"].attrs["reconstruction_settings"]
    assert [e.done for e in events] == [0, 2, 4, 5] and all(e.total == 5 and e.stage == "Saving" for e in events)

    again = reconstruct_plan.save_ram_volume(volume, str(tmp_path / "v.h5"))
    assert again != path and again.endswith("v_1.h5")  # the first file is left alone


def test_section_index_matches_numpy_slicing_on_every_axis():
    a = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    for axis, expected in ((0, a[1]), (1, a[:, 1, :]), (2, a[:, :, 1])):
        np.testing.assert_array_equal(a[section_index(axis, 1)], expected)
    assert axis_labels(1) == ("z", "x") and axis_labels(2) == ("z", "y")


def test_rows_for_export_turns_the_inclusive_save_range_into_a_slice():
    assert reconstruct_plan.rows_for_export(None, 100) == (None, 100)
    assert reconstruct_plan.rows_for_export((0, 99), 100) == (None, 100)
    assert reconstruct_plan.rows_for_export((10, 19), 100) == (slice(10, 20), 10)
    assert reconstruct_plan.rows_for_export((-5, 500), 100) == (None, 100) or reconstruct_plan.rows_for_export((-5, 500), 100) == (slice(0, 100), 100)
    assert reconstruct_plan.rows_for_export((30, 20), 100)[1] == 0  # empty range is reported, not hidden


def test_describe_run_mentions_range_and_output_type():
    all_rows = reconstruct_plan.describe_run(ProjectConfig(path="a"), 100, 100)
    assert "all 100" in all_rows and "Output: float32" in all_rows
    some = reconstruct_plan.describe_run(ProjectConfig(path="a", rows=slice(10, 20), dtype="uint16", dtype_clip_min=0.0, dtype_clip_max=1.0), 100, 10)
    assert "10 to 19 (10 of 100)" in some and "Output: uint16" in some
    unclipped = reconstruct_plan.describe_run(ProjectConfig(path="a", dtype="uint16"), 100, 100)
    assert "float32" in unclipped and "Clip min/max" in unclipped


def test_output_type_defaults_to_float32():
    assert ProjectConfig(path="a").dtype == "float32"


def test_streaming_a_selected_slice_range_writes_only_those_slices(tmp_path):
    pytest.importorskip("algotom")
    from tests.fixtures.realistic_phantom import generate as generate_phantom
    from tests.fixtures.realistic_phantom import write_hdf5

    source = tmp_path / "scan.h5"
    write_hdf5(generate_phantom(
        shape=(10, 32, 32), n_angles=60, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=1,
    ), source)
    rows, n = reconstruct_plan.rows_for_export((3, 6), 10)
    base = ProjectConfig(path=str(source), format="hdf5", storage_mode="disk", center=16.0, tilt_deg=0.0, rows=rows)
    run = PipelineRun(base)
    run.import_data()
    run.complete()
    config, _ = streaming_config(base, str(tmp_path / "sub.h5"))
    streamer = PipelineRun(config)
    streamer.project = run.project
    with h5py.File(streamer.stream(), "r") as f:
        assert f["/volume"].shape == (4, 32, 32) and n == 4 and f["/volume"].dtype == np.float32

    uint_config, note = streaming_config(
        ProjectConfig(path=str(source), format="hdf5", center=16.0, tilt_deg=0.0, rows=rows, dtype="uint16",
                      dtype_clip_min=0.0, dtype_clip_max=0.1), str(tmp_path / "u16.h5"))
    streamer = PipelineRun(uint_config)
    streamer.project = run.project
    with h5py.File(streamer.stream(), "r") as f:
        assert f["/volume"].shape == (4, 32, 32) and f["/volume"].dtype == np.uint16 and note == ""


# -- speed / time remaining -------------------------------------------------------


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _tracker():
    from gui.progress_stats import ProgressTracker

    clock = _Clock()
    return ProgressTracker(clock), clock


def _event(done, total=100, stage="Reconstructing", project="p"):
    from orchestration import ProgressEvent

    return ProgressEvent(project, stage, done, total)


def test_speed_and_time_remaining_follow_the_slices_done():
    tracker, clock = _tracker()
    assert tracker.update(_event(0)).speed is None  # nothing measurable yet
    clock.t += 5.0
    stats = tracker.update(_event(10))
    assert stats.speed == pytest.approx(2.0)
    assert stats.remaining_s == pytest.approx(45.0)  # 90 slices left at 2 per second
    assert stats.elapsed_s == pytest.approx(5.0) and not stats.finished


def test_speed_tracks_recent_progress_not_the_whole_run():
    tracker, clock = _tracker()
    tracker.update(_event(0))
    for done in range(10, 100, 10):  # fast start: 10 slices per second
        clock.t += 1.0
        tracker.update(_event(done))
    for done in range(100, 160, 10):  # then it slows to 1 slice per second
        clock.t += 10.0
        stats = tracker.update(_event(done, total=200))
    assert stats.speed == pytest.approx(1.0)


def test_repeated_events_at_the_same_count_do_not_flatten_the_speed():
    tracker, clock = _tracker()
    tracker.update(_event(0))
    clock.t += 2.0
    tracker.update(_event(4))
    for stage in ("Normalizing projections", "Reconstructing", "Writing"):
        clock.t += 1.0
        stats = tracker.update(_event(4, stage=stage))
    clock.t += 1.0
    stats = tracker.update(_event(8))
    assert stats.speed == pytest.approx(8 / 6.0)  # 8 slices over the 6 seconds that passed
    # (the repeated-count events carry no new information and are not samples)


def test_finished_run_reports_total_time_and_average_speed():
    tracker, clock = _tracker()
    tracker.update(_event(0, total=50))
    clock.t += 25.0
    stats = tracker.update(_event(50, total=50, stage="Done"))
    assert stats.finished and stats.elapsed_s == pytest.approx(25.0)
    assert stats.average_speed == pytest.approx(2.0) and stats.remaining_s is None
    clock.t += 1000.0
    assert tracker.stats().elapsed_s == pytest.approx(25.0)  # a finished run's length stays fixed


def test_a_new_run_starts_the_timer_over():
    tracker, clock = _tracker()
    tracker.update(_event(0, total=10))
    clock.t += 10.0
    tracker.update(_event(10, total=10, stage="Done"))
    clock.t += 500.0
    stats = tracker.update(_event(0, total=10, project="other"))
    assert stats.elapsed_s == 0.0 and not stats.finished and stats.speed is None


def test_format_duration():
    from gui.progress_stats import format_duration

    assert format_duration(3.14) == "3.1 s"
    assert format_duration(65) == "1 min 05 s"
    assert format_duration(3725) == "1 h 02 min 05 s"
    assert format_duration(-1) == "0.0 s"
