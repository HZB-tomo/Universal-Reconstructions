"""
validate_config(): a cheap dry-run of one project, for checking many
configs before committing to a batch run.

run_batch(): runs every config independently. Validation is advisory
only -- it never skips a project. Every project gets a real attempt,
and any failure (predicted by validation or not) is caught and
recorded without stopping the rest of the batch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from orchestration.config import PipelineStageError, ProjectConfig
from orchestration.pipeline import PipelineResult, PipelineRun


def validate_config(config: ProjectConfig) -> list[str]:
    """
    Dry-runs import, completion, preprocessing, and calibration --
    the stages most likely to reveal a genuine config/data problem --
    without reconstructing or exporting. Reconstruction is the
    dominant cost in a real run, and a systemic reconstruction problem
    (bad backend name, unsupported geometry) surfaces immediately on
    whichever project runs first anyway, so it isn't worth re-checking
    for every project up front.

    Returns a list of problem strings; empty means nothing was caught
    here (not a guarantee the real run will succeed, just that these
    particular checks passed).
    """
    run = PipelineRun(config, interactive=False)
    try:
        run.import_data()
        run.complete()
        run.preprocess()
        run.calibrate()
    except PipelineStageError as e:
        return [str(e)]
    except Exception as e:
        return [f"unexpected error: {e}"]
    finally:
        if run.project is not None:
            run.project.close()
    return []


@dataclass
class BatchReport:
    results: list = field(default_factory=list)                 # list[PipelineResult]
    validation_problems: dict = field(default_factory=dict)      # project_id -> list[str]

    @property
    def succeeded(self) -> list:
        return [r for r in self.results if r.success]

    @property
    def failed(self) -> list:
        return [r for r in self.results if not r.success]


def run_batch(configs: list, progress=None) -> BatchReport:
    validation_problems = {}
    for config in configs:
        problems = validate_config(config)
        if problems:
            validation_problems[config.resolved_project_id()] = problems

    results = []
    for config in configs:
        run = PipelineRun(config, interactive=False, progress=progress)
        try:
            result = run.run_all()
        except Exception as e:
            # run_all() already catches PipelineStageError itself; this
            # catches everything else (a library bug, a corrupted file,
            # an OOM) so one project's unanticipated failure can never
            # take down the rest of the batch.
            result = PipelineResult(
                project_id=run.project_id, config=config, success=False,
                error=PipelineStageError("run_batch", str(e)), log=list(run.log),
            )
        results.append(result)

    return BatchReport(results=results, validation_problems=validation_problems)
