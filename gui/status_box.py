"""
StatusBox: an always-visible readout of what the pipeline is doing --
project, stage, device (CPU/GPU) and a slices-done progress bar. Sits at
the bottom of the project panel; fed ProgressEvents (see
orchestration/progress.py) via on_progress().
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QGroupBox, QLabel, QProgressBar, QVBoxLayout

from orchestration import ProgressEvent

_FINISHED_STAGES = ("Done", "Failed")


class StatusBox(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Status", parent)
        self.project_label = QLabel(self)
        self.stage_label = QLabel(self)
        self.device_label = QLabel(self)
        self.count_label = QLabel(self)
        self.bar = QProgressBar(self)

        layout = QVBoxLayout(self)
        for widget in (self.project_label, self.stage_label, self.device_label, self.bar, self.count_label):
            layout.addWidget(widget)
        self.reset()

    def reset(self) -> None:
        self.project_label.setText("Project: -")
        self.stage_label.setText("Stage: idle")
        self.device_label.setText("Device: -")
        self.count_label.setText("")
        self._set_bar(None, None)

    def _set_bar(self, done: Optional[int], total: Optional[int], busy: bool = False) -> None:
        if total:
            self.bar.setRange(0, total)
            self.bar.setValue(min(done or 0, total))
            self.bar.setFormat("%p%")
        elif busy:
            self.bar.setRange(0, 0)  # indeterminate: running, but no slice count for this stage
        else:
            self.bar.setRange(0, 1)
            self.bar.setValue(0)
            self.bar.setFormat("")

    def on_progress(self, event: ProgressEvent) -> None:
        finished = event.stage in _FINISHED_STAGES
        self.project_label.setText(f"Project: {event.project_id}")
        self.stage_label.setText(f"Stage: {event.stage}" + (f" -- {event.message}" if event.message else ""))
        if event.device is not None:
            self.device_label.setText(f"Device: {event.device.upper()}")
        if event.total:
            done = event.done or 0
            verb = "saved" if event.stage == "Saving" else "reconstructed"
            self.count_label.setText(f"{done} / {event.total} slices {verb}, {event.total - done} left")
        elif finished:
            self.count_label.setText("")
        else:
            self.count_label.setText("slice count not available for this stage")
        if finished and event.stage == "Done" and event.total:
            self._set_bar(event.total, event.total)
            self.count_label.setText(f"{event.total} / {event.total} slices {'saved' if 'saved' in event.message else 'reconstructed'}, 0 left")
        else:
            self._set_bar(event.done, event.total, busy=not finished)
