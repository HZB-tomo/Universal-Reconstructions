"""
Tests for orchestration/cli.py: the `tomo` command's four subcommands,
called directly through main(argv) (not a subprocess) for speed --
exercises the same argument parsing and dispatch a real invocation
would.
"""
import pytest

from orchestration.cli import main
from orchestration.config_ini import load_config
from tests.fixtures.realistic_phantom import generate, write_hdf5

algotom = pytest.importorskip("algotom", reason="algotom not installed -- these tests need the algotom_fbp backend")


def _write_phantom_hdf5(tmp_path, name="scan.h5"):
    phantom = generate(
        shape=(8, 64, 64), n_angles=120, cor_deviation=0.0, tilt_deg=0.0,
        n_scratches=0, n_defective_pixels=0, inhomogeneity_strength=0.0, noise_level=0.0, seed=1,
    )
    path = tmp_path / name
    write_hdf5(phantom, path)
    return path


def _write_single_config(tmp_path, data_path, export_path):
    config_path = tmp_path / "config.ini"
    config_path.write_text(
        "[import]\n"
        f"path = {data_path}\n"
        "format = hdf5\n"
        "\n"
        "[export]\n"
        "format = hdf5\n"
        f"path = {export_path}\n"
    )
    return config_path


def test_init_config_writes_a_loadable_single_project_template(tmp_path, capsys):
    out = tmp_path / "template.ini"

    code = main(["init-config", str(out)])

    assert code == 0
    assert out.exists()
    config = load_config(out)
    assert config.reconstruction_backend is None            # blank in the template: the default for the geometry
    assert "wrote" in capsys.readouterr().out


def test_init_config_batch_writes_a_loadable_batch_template(tmp_path):
    out = tmp_path / "batch_template.ini"

    code = main(["init-config", str(out), "--batch"])

    assert code == 0
    configs = load_config(out)
    assert "project_a" in configs


def test_run_succeeds_and_prints_the_log(tmp_path, capsys):
    data_path = _write_phantom_hdf5(tmp_path)
    export_path = tmp_path / "out.h5"
    config_path = _write_single_config(tmp_path, data_path, export_path)

    code = main(["run", str(config_path)])

    assert code == 0
    out = capsys.readouterr().out
    assert "reconstructed volume" in out
    assert f"done: {export_path}" in out
    assert export_path.exists()


def test_run_exits_nonzero_and_reports_the_error_on_failure(tmp_path, capsys):
    config_path = tmp_path / "config.ini"
    config_path.write_text("[import]\npath = does_not_exist.h5\nformat = hdf5\n")

    code = main(["run", str(config_path)])

    assert code == 1
    assert "FAILED" in capsys.readouterr().err


def test_run_rejects_a_batch_config(tmp_path, capsys):
    config_path = tmp_path / "batch.ini"
    config_path.write_text("[scanA:import]\npath = x.h5\n")

    code = main(["run", str(config_path)])

    assert code == 1
    assert "tomo batch" in capsys.readouterr().err


def test_validate_reports_ok_for_a_good_config(tmp_path, capsys):
    data_path = _write_phantom_hdf5(tmp_path)
    config_path = _write_single_config(tmp_path, data_path, tmp_path / "out.h5")

    code = main(["validate", str(config_path)])

    assert code == 0
    assert "OK" in capsys.readouterr().out


def test_validate_reports_problems_and_exits_nonzero(tmp_path, capsys):
    config_path = tmp_path / "config.ini"
    config_path.write_text("[import]\npath = does_not_exist.h5\nformat = hdf5\n")

    code = main(["validate", str(config_path)])

    assert code == 1
    assert "does_not_exist" in capsys.readouterr().out


def test_batch_runs_every_project_and_exits_zero_when_all_succeed(tmp_path, capsys):
    data_a = _write_phantom_hdf5(tmp_path, "a.h5")
    data_b = _write_phantom_hdf5(tmp_path, "b.h5")
    config_path = tmp_path / "batch.ini"
    config_path.write_text(
        "[DEFAULT:reconstruction]\n"
        "backend = algotom_fbp\n"
        "\n"
        f"[a:import]\npath = {data_a}\nformat = hdf5\n\n"
        f"[a:export]\nformat = hdf5\npath = {tmp_path / 'a_out.h5'}\n\n"
        f"[b:import]\npath = {data_b}\nformat = hdf5\n\n"
        f"[b:export]\nformat = hdf5\npath = {tmp_path / 'b_out.h5'}\n"
    )

    code = main(["batch", str(config_path)])

    out = capsys.readouterr().out
    assert code == 0
    assert "a: OK" in out
    assert "b: OK" in out


def test_batch_exits_nonzero_when_one_project_fails_but_still_runs_the_rest(tmp_path, capsys):
    data_a = _write_phantom_hdf5(tmp_path, "a.h5")
    config_path = tmp_path / "batch.ini"
    config_path.write_text(
        f"[a:import]\npath = {data_a}\nformat = hdf5\n\n"
        f"[a:export]\nformat = hdf5\npath = {tmp_path / 'a_out.h5'}\n\n"
        "[b:import]\npath = does_not_exist.h5\nformat = hdf5\n"
    )

    code = main(["batch", str(config_path)])

    out = capsys.readouterr().out
    assert code == 1
    assert "a: OK" in out
    assert "b: FAILED" in out


def test_batch_rejects_a_single_project_config(tmp_path, capsys):
    config_path = tmp_path / "config.ini"
    config_path.write_text("[import]\npath = x.h5\n")

    code = main(["batch", str(config_path)])

    assert code == 1
    assert "tomo run" in capsys.readouterr().err
