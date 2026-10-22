"""PreviewEngine: cached single-slice previews for trying out centers of rotation."""
import copy

import numpy as np
import pytest

from gui.preview_engine import PreviewEngine
from orchestration import PipelineRun, ProjectConfig

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
    import gui.preview_engine as pe

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
    assert seen == ["algotom_fbp", "algotom_fbp"]
