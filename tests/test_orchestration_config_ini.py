"""
Tests for orchestration/config_ini.py: load_config() (single-project
and batch grammar), save_config(), write_config_template(), and
resolve_config()'s explicit-then-sidecar precedence.
"""
import pytest

from orchestration.config import ProjectConfig
from orchestration.config_ini import load_config, resolve_config, save_config, write_config_template


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
    assert config.reconstruction_backend == "algotom_fbp"   # named in this file
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

    assert config.reconstruction_backend is None            # blank: the default for the project's geometry


def test_write_config_template_batch_produces_a_loadable_batch_config(tmp_path):
    path = tmp_path / "batch_template.ini"
    write_config_template(path, batch=True)

    configs = load_config(path)  # should not raise

    assert "project_a" in configs
    assert configs["project_a"].reconstruction_backend is None


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


def test_save_config_round_trips_through_load_config(tmp_path):
    config = ProjectConfig(
        project_id="scanA", path="data/scanA.h5", format="hdf5", storage_mode="disk",
        completion_answers={"geometry_type": "parallel"},
        rows=slice(4, 100), zinger_removal=True, zinger_threshold=0.1, zinger_size=3,
        ring_correction=True, ring_method="large_stripe",
        calibration_backend="algotom", center=56.0, tilt_deg=0.25,
        reconstruction_backend="algotom_fbp", device="cpu", rotate_deg=12.5,
        crop=((4, 8), (0, 96), (0, 96)), dtype="uint16", dtype_clip_min=0.0, dtype_clip_max=0.05,
        export_format="hdf5", export_path="out/volume.h5", stream=True, stream_chunk_rows=16,
    )
    path = tmp_path / "scanA_config.ini"

    save_config(config, path)
    reloaded = load_config(path)

    for field in (
        "project_id", "path", "format", "storage_mode", "completion_answers",
        "zinger_removal", "zinger_threshold", "zinger_size", "ring_correction", "ring_method",
        "calibration_backend", "center", "tilt_deg",
        "reconstruction_backend", "device", "rotate_deg",
        "crop", "dtype", "dtype_clip_min", "dtype_clip_max",
        "export_format", "export_path", "stream", "stream_chunk_rows",
    ):
        assert getattr(reloaded, field) == getattr(config, field), field
    assert reloaded.rows.start == 4 and reloaded.rows.stop == 100


def test_save_config_omits_none_fields(tmp_path):
    config = ProjectConfig(path="scan.h5", center=None, crop=None, device=None)
    path = tmp_path / "scan.ini"

    save_config(config, path)

    text = path.read_text()
    assert "center" not in text
    assert "crop" not in text
    assert "device" not in text


def test_save_config_skips_reconstruction_and_ring_options(tmp_path):
    # not in the INI grammar at all (see FIELD_SPECS's module docstring)
    # -- confirmed by save_config() not raising even when they're set.
    config = ProjectConfig(
        path="scan.h5",
        reconstruction_options={"num_iter": 50},
        ring_options={"sigma": 3},
    )
    path = tmp_path / "scan.ini"

    save_config(config, path)

    text = path.read_text()
    assert "num_iter" not in text
    assert "sigma" not in text


def test_save_config_writes_completion_answers_section(tmp_path):
    config = ProjectConfig(path="scan.h5", completion_answers={"geometry_type": "parallel", "sample_name": "foo"})
    path = tmp_path / "scan.ini"

    save_config(config, path)
    reloaded = load_config(path)

    assert reloaded.completion_answers == {"geometry_type": "parallel", "sample_name": "foo"}


def test_resolve_config_returns_none_when_nothing_found(tmp_path):
    data_path = tmp_path / "scan.h5"
    data_path.write_bytes(b"")

    assert resolve_config(data_path) is None


def test_projections_normalized_ini_key(tmp_path):
    path = tmp_path / "scan.ini"
    path.write_text("[import]\npath = x.h5\n\n[preprocessing]\nprojections_normalized = true\n")
    assert load_config(path).projections_normalized is True
    path.write_text("[import]\npath = x.h5\n")
    assert load_config(path).projections_normalized is False
