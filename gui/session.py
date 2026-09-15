"""
ProjectSession: one imported project as the GUI tracks it. Pairs a
ProjectConfig with the PipelineRun driving it through the backend, plus
GUI-only state that belongs in neither of those.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from orchestration import PipelineRun, ProjectConfig


@dataclass
class ProjectSession:
    config: ProjectConfig
    run: Optional[PipelineRun] = None
    checked: bool = False
    # None until AdjustmentPanel knows this project's row count and
    # resolves it to the middle row -- see _refresh_from_session().
    preview_row: Optional[int] = None
    error: Optional[str] = None

    @property
    def project_id(self) -> str:
        return self.config.resolved_project_id()

    @property
    def imported(self) -> bool:
        return self.run is not None and self.run.project is not None


def unique_project_id(candidate: str, existing: set) -> str:
    """
    ProjectConfig.resolved_project_id() is just the path stem, so two
    files named the same in different folders collide. Disambiguates
    `candidate` against `existing` (a set of ids already in use in the
    project panel).
    """
    if candidate not in existing:
        return candidate
    n = 2
    while f"{candidate}_{n}" in existing:
        n += 1
    return f"{candidate}_{n}"
