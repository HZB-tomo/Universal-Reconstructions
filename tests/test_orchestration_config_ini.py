"""
Tests for orchestration/config_ini.py: load_config() (single-project
and batch grammar), write_config_template(), and resolve_config()'s
explicit-then-sidecar precedence.
"""
import pytest

from orchestration.config_ini import load_config, resolve_config, write_config_template


def test_load_config_single_project_reads_every_section(tmp_path):
    path = tmp_path / "scan.ini"
    path.write_text(
        "[import]\n"
        "path = data/scanA.h5\n"
        "format = hdf5\n"
        "\n"
        "[completion]\n"
        "geometry_type = parallel\n"
        "\n"
        "[calibration]\n"
        "center = 56.0\n"
        "tilt_deg =\n"
        "\n"
        "[reconstruction]\n"
        "backend = algotom_fbp\n"
        "\n"
        "[postprocess]\n"
        "dtype = uint16\n"
        "\n"
        "[export]\n"
        "format = hdf5\n"
        "path = out/volume.h5\n"
    )

    config = load_config(path)

    assert config.path == "data/scanA.h5"
    assert config.format == "hdf5"
    assert config.completion_answers == {"geometry_type": "parallel"}
    assert config.center == 56.0
    assert config.tilt_deg is None  # blank -> stays the dataclass default
    assert config.reconstruction_backend == "algotom_fbp"
    assert config.dtype == "uint16"
    assert config.export_format == "hdf5"
    assert config.export_path == "out/volume.h5"


def test_load_config_strips_an_inline_comment(tmp_path):
    path = tmp_path / "scan.ini"
    path.write_text("[calibration]\ncenter = 56.0   # from last week's calibration\n")

    config = load_config(path)

    assert config.center == 56.0


def test_load_config_blank_value_falls_back_to_the_dataclass_default(tmp_path):
    path = tmp_path / "scan.ini"
    path.write_text("[postprocess]\ndtype =\n")

    config = load_config(path)

    assert config.dtype == "float32"


def test_load_config_crop_parses_six_ints(tmp_path):
    path = tmp_path / "scan.ini"
    path.write_text("[postprocess]\ncrop = 4,8,0,96,0,96\n")

    config = load_config(path)

    assert config.crop == ((4, 8), (0, 96), (0, 96))


def test_load_config_rejects_a_bad_value_with_a_clear_message(tmp_path):
    path = tmp_path / "scan.ini"
    path.write_text("[import]\nformat = bmp\n")

    with pytest.raises(ValueError, match="bmp"):
        load_config(path)


def test_load_config_collects_every_problem_not_just_the_first(tmp_path):
    path = tmp_path / "scan.ini"
    path.write_text("[import]\nformat = bmp\n\n[postprocess]\ndtype = int8\n")

    with pytest.raises(ValueError) as exc_info:
        load_config(path)

    message = str(exc_info.value)
    assert "bmp" in message
    assert "int8" in message


def test_load_config_batch_detects_project_scoped_sections(tmp_path):
    path = tmp_path / "batch.ini"
    path.write_text(
        "[DEFAULT:reconstruction]\n"
        "backend = algotom_fbp\n"
        "\n"
        "[scanA:import]\n"
        "path = data/scanA.h5\n"
        "\n"
        "[scanA:calibration]\n"
        "center = 56.0\n"
        "\n"
        "[scanB:import]\n"
        "path = data/scanB.h5\n"
    )

    configs = load_config(path)

    assert set(configs) == {"scanA", "scanB"}
    assert configs["scanA"].path == "data/scanA.h5"
    assert configs["scanA"].center == 56.0
    assert configs["scanA"].reconstruction_backend == "algotom_fbp"  # from DEFAULT:reconstruction
    assert configs["scanB"].reconstruction_backend == "algotom_fbp"  # inherited too
    assert configs["scanB"].center is None  # scanB never set it


def test_load_config_batch_project_section_overrides_default(tmp_path):
    path = tmp_path / "batch.ini"
    path.write_text(
        "[DEFAULT:reconstruction]\n"
        "backend = algotom_fbp\n"
        "\n"
        "[scanA:import]\n"
        "path = data/scanA.h5\n"
        "\n"
        "[scanA:reconstruction]\n"
        "backend = tomopy_fbp\n"
    )

    configs = load_config(path)

    assert configs["scanA"].reconstruction_backend == "tomopy_fbp"


def test_load_config_batch_default_does_not_leak_across_stages(tmp_path):
    """[DEFAULT:import]'s path must never apply to an [export] section --
    the whole reason batch defaults are scoped per-stage rather than
    using configparser's native cross-section [DEFAULT]."""
    path = tmp_path / "batch.ini"
    path.write_text(
        "[DEFAULT:import]\n"
        "path = data/shared.h5\n"
        "\n"
        "[scanA:import]\n"
        "path = data/scanA.h5\n"
        "\n"
        "[scanA:export]\n"
        "format = hdf5\n"
    )

    configs = load_config(path)

    assert configs["scanA"].export_path is None  # not accidentally "data/shared.h5"


def test_load_config_batch_project_id_defaults_to_the_section_token(tmp_path):
    path = tmp_path / "batch.ini"
    path.write_text("[scanA:import]\npath = data/scanA.h5\n")

    configs = load_config(path)

    assert configs["scanA"].project_id == "scanA"


def test_write_config_template_produces_a_loadable_config(tmp_path):
    path = tmp_path / "template.ini"
    write_config_template(path)

    config = load_config(path)  # should not raise

    assert config.reconstruction_backend == "algotom_fbp"


def test_resolve_config_prefers_explicit_over_sidecar(tmp_path):
    data_path = tmp_path / "scan.h5"
    data_path.write_bytes(b"")
    sidecar = tmp_path / "scan.h5.tomo_config.ini"
    sidecar.write_text("[import]\npath = from_sidecar.h5\n")
    explicit = tmp_path / "explicit.ini"
    explicit.write_text("[import]\npath = from_explicit.h5\n")

    config = resolve_config(data_path, explicit_path=explicit)

    assert config.path == "from_explicit.h5"


def test_resolve_config_falls_back_to_sidecar(tmp_path):
    data_path = tmp_path / "scan.h5"
    data_path.write_bytes(b"")
    sidecar = tmp_path / "scan.h5.tomo_config.ini"
    sidecar.write_text("[import]\npath = from_sidecar.h5\n")

    config = resolve_config(data_path)

    assert config.path == "from_sidecar.h5"


def test_resolve_config_returns_none_when_nothing_found(tmp_path):
    data_path = tmp_path / "scan.h5"
    data_path.write_bytes(b"")

    assert resolve_config(data_path) is None
