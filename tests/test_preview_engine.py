"""PreviewEngine: cached single-slice previews for trying out centers of rotation."""
import copy

import numpy as np
import pytest

from gui.preview_engine import PreviewEngine
from orchestration import PipelineRun, ProjectConfig
from reconstruction import default_backend

pytest.importorskip("algotom")
from tests.fixtures.realistic_phantom import generate as generate_phantom  # noqa: E402
from tests.fixtures.realistic_phantom import write_hdf5  # noqa: E402


@pytest.fixture
def project_and_config(tmp_path):
    path = tmp_path / "scan.h5"
    write_hdf5(generate_phantom(
        shape=(6, 32, 32), n_angles=60, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=1,
    ), path)
    config = ProjectConfig(path=str(path), format="hdf5", storage_mode="disk", center=16.0, tilt_deg=0.0)
    run = PipelineRun(config)
    run.import_data()
    run.complete()
    yield run.project, config
    run.project.close()


def test_second_request_for_the_same_center_is_a_cache_hit(project_and_config):
    project, config = project_and_config
    engine = PreviewEngine()
    first = engine.compute(project, config, row=3, center=16.0)
    again = engine.compute(project, config, row=3, center=16.0)
    assert first.shape == (32, 32) and first.dtype == np.float32
    assert again is first and engine.reconstructions == 1


def test_a_different_center_row_or_setting_is_a_miss(project_and_config):
    project, config = project_and_config
    engine = PreviewEngine()
    engine.compute(project, config, 3, 16.0)
    engine.compute(project, config, 3, 16.5)
    engine.compute(project, config, 2, 16.0)
    other = copy.deepcopy(config)
    other.reconstruction_backend = "algotom_bpf"
    engine.compute(project, other, 3, 16.0)
    assert engine.reconstructions == 4
    assert engine.cached(project, config, 3, 16.0) is not None
    assert engine.cached(project, config, 4, 16.0) is None


def test_the_sinogram_is_prepared_once_per_row_however_many_centers(project_and_config, monkeypatch):
    import orchestration.preprocess as pe

    project, config = project_and_config
    calls = []
    original = pe.normalize_rows
    monkeypatch.setattr(pe, "normalize_rows", lambda *a, **k: calls.append(1) or original(*a, **k))
    engine = PreviewEngine()
    for center in (15.0, 15.5, 16.0, 16.5):
        engine.compute(project, config, 3, center)
    assert len(calls) == 1


def test_preview_matches_the_pipeline_for_the_same_slice(project_and_config):
    project, config = project_and_config
    config.rows = slice(3, 4)
    run = PipelineRun(config)
    run.project = project
    run.preprocess()
    run.calibrate()
    expected = run.reconstruct()[0]
    image = PreviewEngine().compute(project, config, 3, 16.0)
    np.testing.assert_allclose(image, expected, rtol=1e-5, atol=1e-6)


def test_cache_is_bounded_and_forgets_the_least_recently_used(project_and_config):
    project, config = project_and_config
    one_image = 32 * 32 * 4
    engine = PreviewEngine(max_bytes=3 * one_image)
    for center in (14.0, 15.0, 16.0):
        engine.compute(project, config, 3, center)
    engine.cached(project, config, 3, 14.0)          # touch the oldest: now 15.0 is the least recently used
    engine.compute(project, config, 3, 17.0)
    assert len(engine) == 3
    assert engine.cached(project, config, 3, 15.0) is None
    assert engine.cached(project, config, 3, 14.0) is not None


def test_prefetch_fills_neighbours_skips_known_ones_and_can_be_cancelled(project_and_config):
    project, config = project_and_config
    engine = PreviewEngine()
    engine.compute(project, config, 3, 16.0)
    done = engine.prefetch(project, config, 3, [16.0, 17.0, 15.0, 18.0, 14.0])
    assert done == 4 and engine.reconstructions == 5
    assert all(engine.cached(project, config, 3, c) is not None for c in (14, 15, 16, 17, 18))

    state = {"n": 0}

    def cancel_after_one():
        state["n"] += 1
        return state["n"] > 1

    engine2 = PreviewEngine()
    assert engine2.prefetch(project, config, 3, [10.0, 11.0, 12.0], cancelled=cancel_after_one) == 1


def test_prefetch_uses_a_snapshot_of_the_config(project_and_config):
    project, config = project_and_config
    engine = PreviewEngine()
    seen = []
    original = engine.compute

    def spy(project, cfg, row, center):
        seen.append(cfg.reconstruction_backend)
        config.reconstruction_backend = "algotom_bpf"  # the GUI edits the live config mid-prefetch
        return original(project, cfg, row, center)

    engine.compute = spy
    engine.prefetch(project, config, 3, [15.0, 16.0])
    assert seen == [default_backend("parallel")] * 2


# -- scanning the center of rotation ------------------------------------------------------------------------

def test_scan_finds_the_center_with_a_sinogram_metric_and_with_a_slice_metric(project_and_config):
    from calibration import center_grid

    project, config = project_and_config
    engine = PreviewEngine()
    centers = center_grid(32, -6.5, 6.5, 27)                          # the phantom's axis is at 15.5
    quick = engine.scan(project, config, 3, "vo", centers)
    assert quick.best == pytest.approx(15.5, abs=1.0) and quick.approximate == ""
    assert engine.reconstructions == 0                                 # a sinogram metric reconstructs nothing
    sharp = engine.scan(project, config, 3, "sharpness", centers)
    assert sharp.best == pytest.approx(15.5, abs=1.0)
    assert engine.reconstructions == 27
    assert engine.cached(project, config, 3, float(sharp.centers[5])) is not None     # so scrubbing afterwards is instant


def test_scan_reports_progress_and_can_be_cancelled(project_and_config):
    from calibration import ScanCancelled, center_grid

    project, config = project_and_config
    engine = PreviewEngine()
    seen = []
    engine.scan(project, config, 3, "sharpness", center_grid(32, -2, 2, 5), progress=lambda d, t: seen.append((d, t)))
    assert seen == [(k, 5) for k in range(1, 6)]
    stop_after = iter([False, False, True])
    with pytest.raises(ScanCancelled):
        engine.scan(project, config, 3, "sharpness", center_grid(32, -3, 3, 9) + 0.01, cancelled=lambda: next(stop_after, True))


def test_scan_does_not_touch_the_callers_config(project_and_config):
    from calibration import center_grid

    project, config = project_and_config
    before = copy.deepcopy(config)
    PreviewEngine().scan(project, config, 3, "sharpness", center_grid(32, -1, 1, 3))
    assert config == before


def test_auto_center_is_one_search_and_shows_the_curve_the_search_evaluated(project_and_config, monkeypatch):
    import calibration.backends.algotom_cor as cor

    project, config = project_and_config
    calls = []
    original = cor.AlgotomCorBackend.search_centers
    monkeypatch.setattr(cor.AlgotomCorBackend, "search_centers", lambda self, *a, **k: calls.append(1) or original(self, *a, **k))
    engine = PreviewEngine()
    search, scan, window = engine.auto_center(project, config, 3)
    assert calls == [1] and engine.reconstructions == 0                  # nothing computed besides the search itself
    assert search.best == pytest.approx(15.5, abs=0.75)
    assert scan.best == search.best                                       # the search's own refined answer, not a grid point
    assert scan.metric == "vo" and not scan.at_edge
    assert len(scan.centers) == 8 and scan.centers[0] == 12.0 and scan.centers[-1] == 19.0   # every column of 15.5 +/- 3.2 (a tenth of 32)
    assert window == (12.0 - 15.5, 19.0 - 15.5)


def test_auto_center_searches_the_window_it_is_given_and_says_when_the_answer_is_at_its_edge(project_and_config):
    project, config = project_and_config
    engine = PreviewEngine()
    search, scan, shown = engine.auto_center(project, config, 3, window=(-8.0, 8.0))
    assert scan.centers[0] == 7.0 and scan.centers[-1] == 24.0 and shown == (-8.5, 8.5)      # 15.5 +/- 8, rounded out to whole columns
    assert search.best == pytest.approx(15.5, abs=0.75) and not search.at_edge
    narrow = engine.auto_center(project, config, 3, window=(2.0, 6.0))[0]                    # the axis (shift 0) is outside
    assert narrow.at_edge
