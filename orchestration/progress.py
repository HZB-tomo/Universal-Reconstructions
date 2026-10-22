"""
ProgressEvent: what a running PipelineRun reports to an optional
callback (PipelineRun(progress=...), run_batch(progress=...)) -- the
stage it is in, how many slices are done, and the device in use. The
callback runs on whichever thread runs the pipeline, so a GUI must
hand the event over to its own thread (see gui/workers.py's
ProgressRelay).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProgressEvent:
    project_id: str
    stage: str                      # e.g. "Importing", "Reconstructing", "Writing", "Done", "Failed"
    done: Optional[int] = None      # slices finished so far; None = not a slice-counted stage
    total: Optional[int] = None     # slices in this run; None = unknown/indeterminate
    device: Optional[str] = None    # "cpu" or "gpu" once a reconstruction device is known
    message: str = ""
