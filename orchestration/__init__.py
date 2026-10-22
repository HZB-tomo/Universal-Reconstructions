"""
Drives one or many projects through the full pipeline (import,
completion, preprocessing, calibration, reconstruction, postprocess,
export), configured by a ProjectConfig -- built directly, loaded from
an INI file, or both.

Public API:
    ProjectConfig                     -> one project's settings, see config.py
    PipelineStageError                -> raised by any PipelineRun stage on failure
    PipelineRun                       -> drives one project; independently callable
                                          stage methods, plus run_all() to chain them
    PipelineResult                    -> run_all()'s return value
    load_config, save_config, write_config_template, resolve_config  -> INI I/O, see config_ini.py
    validate_config                   -> cheap dry-run of one config, see batch.py
    run_batch, BatchReport            -> run every config in a list independently
    ram_check                         -> RAM estimation/chunk sizing, see ram_check.py
"""
from orchestration import ram_check  # noqa: F401
from orchestration.batch import BatchReport, run_batch, validate_config
from orchestration.config import PipelineStageError, ProjectConfig
from orchestration.config_ini import load_config, resolve_config, save_config, write_config_template
from orchestration.pipeline import PipelineResult, PipelineRun
from orchestration.progress import ProgressEvent

__all__ = [
    "ProjectConfig", "PipelineStageError", "PipelineRun", "PipelineResult",
    "load_config", "save_config", "write_config_template", "resolve_config",
    "validate_config", "run_batch", "BatchReport", "ProgressEvent",
]
