"""Backend availability (`requires`) and the registry listing the GUI is built from."""
import numpy as np
import pytest

import reconstruction.backend as backend_module
from reconstruction.backend import ReconstructionBackend, get_backend, list_backends, register_backend


@pytest.fixture
def scratch_registry(monkeypatch):
    """A copy of the registry that a test can add fake backends to."""
    monkeypatch.setattr(backend_module, "_REGISTRY", dict(backend_module._REGISTRY))
    return backend_module._REGISTRY


def _fake(name, requires=()):
    def reconstruct(self, sinograms, angles, center, geometry, device=None, **options):
        return np.zeros((sinograms.shape[1], sinograms.shape[2], sinograms.shape[2]))

    cls = type("Fake", (ReconstructionBackend,), {
        "name": name, "requires": requires, "supported_geometries": {"parallel"}, "reconstruct": reconstruct,
    })
    register_backend(cls)       # instantiates it and adds it to the (scratch) registry
    return cls


def test_a_backend_with_no_requirements_is_available(scratch_registry):
    _fake("plain")
    assert get_backend("plain").is_available()
    assert get_backend("plain").missing_requirements() == []


def test_missing_packages_make_a_backend_unavailable_and_are_named(scratch_registry):
    _fake("needs_things", requires=("numpy", "no_such_module_for_urct_tests"))
    backend = get_backend("needs_things")
    assert not backend.is_available()
    assert backend.missing_requirements() == ["no_such_module_for_urct_tests"]


def test_list_backends_hides_unavailable_ones_unless_asked(scratch_registry):
    _fake("needs_things", requires=("no_such_module_for_urct_tests",))
    assert "needs_things" not in list_backends()
    assert "needs_things" in list_backends(available_only=False)


def test_list_backends_uses_the_display_order_then_alphabetical(scratch_registry):
    backend_module._REGISTRY.clear()
    for name in ("zeta_custom", "algotom_bpf", "alpha_custom", "gridrec", "algotom_fbp"):
        _fake(name)
    assert list_backends() == ["algotom_fbp", "gridrec", "algotom_bpf", "alpha_custom", "zeta_custom"]


def test_a_user_backend_shows_up_without_any_gui_change(scratch_registry):
    _fake("my_backend")
    assert "my_backend" in list_backends()


def test_builtin_backends_declare_what_they_need():
    assert "algotom" in get_backend("algotom_fbp").requires
    assert set(get_backend("algotom_gridrec").requires) == {"algotom", "tomopy"}
    for name in ("algotom_bp", "algotom_sirt", "algotom_sart", "algotom_art", "algotom_cgls"):
        assert "astra" in get_backend(name).requires      # subclasses inherit it
    assert "urct_gridrec" in get_backend("gridrec").requires
