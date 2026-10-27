"""The reconstruction filter choice: backend lists, validation, config/INI, pipeline, preview and GUI."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("algotom")
from orchestration import ProjectConfig  # noqa: E402
from orchestration.config_ini import load_config  # noqa: E402
from reconstruction import Geometry, get_backend, list_backends, reconstruct  # noqa: E402
from tests.test_gridrec_backend import GEOMETRY, disk_stack  # noqa: E402


def noisy_stack(seed=0):
    stack, angles = disk_stack()
    rng = np.random.default_rng(seed)
    return np.clip(stack * (1 + 0.03 * rng.standard_normal(stack.shape)), 1e-3, None).astype(np.float32), angles


def noise_level(volume):
    """Standard deviation in a flat part inside the big disk."""
    return float(volume[:, 19:25, 23:29].std())


@pytest.mark.parametrize("name", ["algotom_fbp", "algotom_bpf", "algotom_dfi", "gridrec"])
def test_every_listed_filter_runs_and_ramp_is_the_noisiest(name):
    if name not in list_backends():
        pytest.skip(f"{name} not available")
    stack, angles = noisy_stack()
    backend = get_backend(name)
    noise = {}
    for filt in backend.filters:
        out = reconstruct(stack, angles, 23.5, geometry=GEOMETRY, backend=name, device="cpu", filter_name=filt)
        assert out.shape == (3, 48, 48) and np.isfinite(out).all()
        noise[filt] = noise_level(out)
    assert noise["ramp"] > noise["hann"]
    assert backend.default_filter in backend.filters


def test_unknown_filter_is_rejected_not_silently_turned_into_ramp():
    stack, angles = disk_stack()
    with pytest.raises(ValueError, match="no filter 'nope'"):
        reconstruct(stack, angles, 23.5, geometry=GEOMETRY, backend="algotom_fbp", device="cpu", filter_name="nope")


def test_iterative_backends_declare_no_filters():
    assert get_backend("algotom_sirt").filters == ()


def test_config_merges_the_filter_into_the_options():
    config = ProjectConfig(path="x", reconstruction_backend="algotom_fbp", reconstruction_filter="hamming")
    assert config.backend_options() == {"filter_name": "hamming"}
    assert ProjectConfig(path="x").backend_options() == {}
    assert ProjectConfig(path="x", reconstruction_filter="hamming", reconstruction_options={"filter_name": "ramp"}).backend_options() == {"filter_name": "ramp"}
    sirt = ProjectConfig(path="x", reconstruction_backend="algotom_sirt", reconstruction_filter="hamming")
    assert "filter_name" not in sirt.backend_options()


def test_ini_round_trip(tmp_path):
    ini = tmp_path / "p.ini"
    ini.write_text("[import]\npath = a.h5\n[reconstruction]\nbackend = gridrec\nfilter = shepp-logan\n")
    assert load_config(str(ini)).reconstruction_filter == "shepp-logan"


def test_preview_cache_key_follows_the_filter():
    from gui.preview_engine import reconstruction_key

    a = ProjectConfig(path="x")
    b = ProjectConfig(path="x", reconstruction_filter="ramp")
    assert reconstruction_key(a) != reconstruction_key(b)


def test_gui_filter_combo_follows_the_backend():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from gui.adjustment_panel import BasicSettingsTab

    app = QApplication.instance() or QApplication([])
    tab = BasicSettingsTab()
    args = dict(geometry_type="parallel", angle_span=180.0, angle_span_derived=False, pixel_size=1.0, pixel_size_derived=False,
                slice_index=0, slice_max=10, backend="gridrec", num_iter=50, is_iterative=False, device=None, center=10.0,
                tilt_deg=0.0, rotate_deg=0.0, width=20)
    seen = []
    tab.filter_changed.connect(seen.append)
    tab.set_values(filters=get_backend("gridrec").filters, filter_name="hamming", **args)
    assert not tab.filter_combo.isHidden() and tab.filter_combo.currentText() == "hamming" and seen == []
    tab.filter_combo.setCurrentText("ramp")
    assert seen == ["ramp"]
    tab.set_values(filters=(), filter_name=None, **{**args, "backend": "algotom_sirt", "is_iterative": True})
    assert tab.filter_combo.isHidden()
