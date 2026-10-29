"""What a project is reconstructed with when nobody chose: gridrec for parallel beam, astra_fdk (GPU) for cone beam."""
import configparser

import h5py
import numpy as np
import pytest

from orchestration import PipelineRun, ProjectConfig, load_config, save_config
from reconstruction import ReconstructionBackend, default_backend, get_backend, list_backends
from reconstruction import backend as backend_module
from reconstruction.backend import DEFAULT_BACKENDS, register_backend
from tests.fixtures.cone_phantom import COLS, N_ANGLES, ROWS, SDD, SOD, analytic_projections, voxel_size_mm


@pytest.fixture
def scratch_registry():
    saved = dict(backend_module._REGISTRY)
    yield
    backend_module._REGISTRY.clear()
    backend_module._REGISTRY.update(saved)


def _fake(name, geometries, available=True):
    cls = type("Fake", (ReconstructionBackend,), {
        "name": name, "supported_geometries": set(geometries),
        "requires": () if available else ("no_such_module_xyz",),
        "reconstruct": lambda self, *a, **k: None,
    })
    register_backend(cls)       # instantiates it and adds it to the (scratch) registry
    return cls


def test_the_preferences_are_gridrec_for_parallel_and_astra_fdk_for_cone():
    assert DEFAULT_BACKENDS["parallel"][0] == "gridrec" and DEFAULT_BACKENDS["cone"][0] == "astra_fdk"
    assert default_backend("parallel") == default_backend(None) == "gridrec"         # installed here
    expected = "astra_fdk" if get_backend("astra_fdk").is_available() else "rtk_fdk"
    assert default_backend("cone") == expected


def test_the_algorithm_list_starts_with_the_defaults():
    assert list_backends(geometry="parallel")[0] == "gridrec"
    assert list_backends(geometry="cone")[:1] == ["astra_fdk"] or not get_backend("astra_fdk").is_available()


def test_a_missing_preferred_backend_falls_back_to_the_next_then_to_whatever_supports_the_geometry(scratch_registry):
    backend_module._REGISTRY.clear()
    _fake("gridrec", {"parallel"}, available=False)
    _fake("algotom_fbp", {"parallel"})
    _fake("zzz_custom", {"parallel"})
    assert default_backend("parallel") == "algotom_fbp"
    backend_module._REGISTRY.clear()
    _fake("gridrec", {"parallel"}, available=False)
    _fake("algotom_fbp", {"parallel"}, available=False)
    _fake("zzz_custom", {"parallel"})
    assert default_backend("parallel") == "zzz_custom"
    assert default_backend("cone") is None                       # nothing supports cone beam here


def test_resolving_sets_an_unset_backend_once_and_leaves_a_chosen_one(scratch_registry):
    config = ProjectConfig(path="x")
    assert config.reconstruction_backend is None
    assert config.resolve_backend("parallel") == "gridrec" == config.reconstruction_backend
    assert config.resolve_backend("cone") == "gridrec"           # already settled: not changed behind anyone's back
    chosen = ProjectConfig(path="x", reconstruction_backend="algotom_sirt")
    assert chosen.resolve_backend("parallel") == "algotom_sirt"
    backend_module._REGISTRY.clear()
    with pytest.raises(ValueError, match="no usable cone-beam"):
        ProjectConfig(path="x").resolve_backend("cone")


def test_an_unset_backend_is_left_out_of_a_saved_config_and_reads_back_unset(tmp_path):
    path = tmp_path / "c.ini"
    save_config(ProjectConfig(path="a.h5", format="hdf5"), path)
    parser = configparser.ConfigParser()
    parser.read(path)
    assert not parser.has_option("reconstruction", "backend")
    assert load_config(path).reconstruction_backend is None
    save_config(ProjectConfig(path="a.h5", format="hdf5", reconstruction_backend="algotom_fbp"), path)
    assert load_config(path).reconstruction_backend == "algotom_fbp"


def test_a_parallel_beam_run_with_no_backend_chosen_uses_gridrec(tmp_path):
    from tests.fixtures.realistic_phantom import generate, write_hdf5

    path = tmp_path / "scan.h5"
    write_hdf5(generate(shape=(4, 32, 32), n_angles=40, cor_deviation=0.0, tilt_deg=0.0, n_scratches=0,
                        n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=1), path)
    config = ProjectConfig(path=str(path), format="hdf5", center=16.0, tilt_deg=0.0, export_path=str(tmp_path / "o.h5"))
    result = PipelineRun(config).run_all()
    assert result.success, result.error
    assert config.reconstruction_backend == "gridrec"
    with h5py.File(result.volume_path, "r") as f:
        assert f["/entry/volume/data"].shape == (4, 32, 32)


def _cone_file(path):
    angles = np.arange(N_ANGLES) * 360.0 / N_ANGLES
    with h5py.File(path, "w") as f:
        f.attrs["beamline"] = "generic_parallel_synchrotron"
        f.create_dataset("/entry/instrument/detector/data", data=np.exp(-analytic_projections()).astype(np.float32))
        f.create_dataset("/entry/instrument/detector/flat_fields", data=np.ones((2, ROWS, COLS), dtype=np.float32))
        f.create_dataset("/entry/instrument/detector/dark_fields", data=np.zeros((2, ROWS, COLS), dtype=np.float32))
        f.create_dataset("/entry/instrument/sample/rotation_angle", data=angles)
        f.create_dataset("/entry/instrument/detector/pixel_size", data=voxel_size_mm() * 1000.0)
    return path


def test_a_cone_beam_run_with_no_backend_chosen_gets_a_cone_backend_not_an_error(tmp_path):
    path = _cone_file(tmp_path / "cone.h5")
    answers = {"geometry_type": "cone", "source_to_detector_distance": SDD, "source_to_object_distance": SOD}
    config = ProjectConfig(path=str(path), format="hdf5", center=33.5, tilt_deg=0.0, completion_answers=answers,
                           export_path=str(tmp_path / "o.h5"))
    result = PipelineRun(config).run_all()
    assert result.success, result.error
    assert config.reconstruction_backend == default_backend("cone") and "cone" in get_backend(config.reconstruction_backend).supported_geometries


def test_a_backend_chosen_for_the_wrong_geometry_is_still_an_error(tmp_path):
    path = _cone_file(tmp_path / "cone.h5")
    answers = {"geometry_type": "cone", "source_to_detector_distance": SDD, "source_to_object_distance": SOD}
    config = ProjectConfig(path=str(path), format="hdf5", center=33.5, tilt_deg=0.0, completion_answers=answers,
                           reconstruction_backend="algotom_fbp", export_path=str(tmp_path / "o.h5"))
    with pytest.raises(ValueError, match="doesn't support"):
        PipelineRun(config).run_all()
    assert config.reconstruction_backend == "algotom_fbp"        # an explicit choice is never replaced
