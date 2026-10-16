"""
Shared pytest setup. Two things happen here:

1. We make sure `import core`, `import data_import`, etc. work when
   pytest is run from anywhere, by adding the repo root to sys.path.

2. We define `synthetic_data`, a fast/small version of the synthetic
   fixture (see tests/fixtures/generate_synthetic.py) that every test
   needing real data can request as a normal pytest fixture argument.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib

from data_import.plugin import clear_registry

# The actual modules holding @register_plugin decorators -- see the
# reload note below. Any NEW vendor module must be added to this import
# AND to the reload() calls below, or it will silently stop being
# registered after the first test that runs.
from data_import.vendors import generic, synthetic, hypothetical_synchrotron, generic_parallel_synchrotron, bamline, ill_neutrons, desy, rx_soln
from tests.fixtures.generate_synthetic import generate


@pytest.fixture(autouse=True)
def _fresh_registry():
    """
    Runs automatically before every single test. Clears the plugin
    registry and re-registers the built-ins, so tests that register
    extra plugins (e.g. to test an ambiguous-match scenario) don't leak
    those plugins into other tests.

    Important: we reload each vendor module directly, NOT the
    `data_import.vendors` package. Reloading a package only re-runs that
    package's own __init__.py -- its `from . import generic` line would
    just re-bind the name from Python's module cache (sys.modules)
    without re-executing generic.py's top-level code, so the
    @register_plugin decorators inside it would never fire again. We
    need the submodules themselves reloaded so those decorators actually
    re-run and re-populate the registry we just cleared.
    """
    clear_registry()
    importlib.reload(generic)
    importlib.reload(synthetic)
    importlib.reload(hypothetical_synchrotron)
    importlib.reload(generic_parallel_synchrotron)
    importlib.reload(bamline)
    yield


@pytest.fixture
def synthetic_data(tmp_path):
    """
    A small, fast synthetic dataset for quick test runs -- NOT the full
    256x256x512 / 800-angle spec (that takes minutes; use generate()
    directly for that, e.g. via generate_synthetic.py's __main__ block).

    Returns a dict with paths to both the hdf5 and tiff versions, plus
    the ground-truth volume/projections/angles so tests can verify
    loaded data actually matches what was written.
    """
    volume, projections, angles = generate(
        tmp_path / "synthetic", shape=(48, 32, 32), n_angles=20, seed=42
    )
    return {
        "dir": tmp_path / "synthetic",
        "hdf5_path": tmp_path / "synthetic" / "synthetic.h5",
        "tiff_path": tmp_path / "synthetic" / "synthetic_data",
        "volume": volume,
        "projections": projections,
        "angles": angles,
    }
