"""
"Import to RAM" must really load the data into memory: RamHandles, no
open file, and no disk reads in the pipeline stages that touch the
projections (normalization, reconstruction). The disk mode is the
control -- it must read from the file on every access.
"""
import h5py
import numpy as np
import pytest

from core.handles import Hdf5DiskHandle, RamHandle
from orchestration import PipelineRun, ProjectConfig
from tests.fixtures.realistic_phantom import generate as generate_phantom
from tests.fixtures.realistic_phantom import write_hdf5

algotom = pytest.importorskip("algotom")


@pytest.fixture
def scan(tmp_path):
    path = tmp_path / "scan.h5"
    write_hdf5(generate_phantom(
        shape=(6, 32, 32), n_angles=60, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=1,
    ), path)
    return path


def _config(path, storage_mode, **kwargs):
    return ProjectConfig(path=str(path), format="hdf5", storage_mode=storage_mode, center=16.0, tilt_deg=0.0, **kwargs)


@pytest.fixture
def read_counter(monkeypatch):
    """Counts h5py dataset reads (every array access goes through __getitem__)."""
    counter = {"reads": 0}
    original = h5py.Dataset.__getitem__

    def counting(self, args):
        counter["reads"] += 1
        return original(self, args)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", counting)
    return counter


def test_ram_import_holds_arrays_in_memory_and_releases_the_file(scan):
    run = PipelineRun(_config(scan, "ram"))
    project = run.import_data()
    assert project.storage_mode == "ram"
    for key, handle in project.data_handles.items():
        assert isinstance(handle, RamHandle), key
        assert handle.storage_mode == "ram"
    assert project._open_resource is None

    # The file is no longer held open: it can be deleted, and the data
    # is still all there.
    scan.unlink()
    assert not scan.exists()
    run.complete()
    normalized = run.preprocess()
    assert normalized.shape[1] == 6 and np.isfinite(normalized).all()


def test_disk_import_keeps_the_file_open_and_reads_from_it(scan):
    run = PipelineRun(_config(scan, "disk"))
    project = run.import_data()
    try:
        assert isinstance(project.data_handles["projections"], Hdf5DiskHandle)
        assert project._open_resource is not None
    finally:
        project.close()


def test_ram_mode_does_no_disk_reads_during_preprocess_and_reconstruct(scan, read_counter):
    run = PipelineRun(_config(scan, "ram"))
    run.import_data()
    run.complete()
    assert read_counter["reads"] > 0  # the import itself did read the file

    read_counter["reads"] = 0
    run.preprocess()
    run.calibrate()
    run.reconstruct()
    assert read_counter["reads"] == 0


def test_disk_mode_reads_from_the_file_during_preprocess(scan, read_counter):
    run = PipelineRun(_config(scan, "disk"))
    run.import_data()
    run.complete()
    read_counter["reads"] = 0
    try:
        run.preprocess()
        assert read_counter["reads"] > 0
    finally:
        run.project.close()


def test_ram_and_disk_give_the_same_reconstruction(scan):
    """algotom's FBP isn't bit-deterministic (see tests/test_streaming.py), so
    RAM-vs-disk must agree at least as well as two RAM runs agree."""
    def run_mode(mode):
        run = PipelineRun(_config(scan, mode))
        run.import_data()
        run.complete()
        run.preprocess()
        run.calibrate()
        volume = run.reconstruct()
        run.project.close()
        return volume

    corr = lambda a, b: np.corrcoef(a.ravel(), b.ravel())[0, 1]
    ram, disk, ram_again = run_mode("ram"), run_mode("disk"), run_mode("ram")
    assert corr(ram, disk) >= corr(ram, ram_again) - 0.03
