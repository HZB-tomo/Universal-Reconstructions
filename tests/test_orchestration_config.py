"""
Tests for orchestration/config.py: ProjectConfig's defaults and
resolved_project_id() fallback.
"""
from orchestration.config import ProjectConfig


def test_resolved_project_id_defaults_to_the_path_stem():
    config = ProjectConfig(path="data/scanA.h5")
    assert config.resolved_project_id() == "scanA"


def test_resolved_project_id_honors_an_explicit_value():
    config = ProjectConfig(path="data/scanA.h5", project_id="my_project")
    assert config.resolved_project_id() == "my_project"


def test_resolved_project_id_works_for_a_tiff_folder_path():
    config = ProjectConfig(path="data/scanB_tiffs", format="tiff")
    assert config.resolved_project_id() == "scanB_tiffs"


def test_defaults_mean_fully_automatic():
    config = ProjectConfig(path="scan.h5")
    assert config.center is None
    assert config.tilt_deg is None
    assert config.dtype == "float32"
    assert config.crop is None
    assert config.completion_answers == {}
