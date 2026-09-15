"""
Runs one callable on a background thread via QThreadPool, so pipeline
stages (import, calibration, reconstruction) never block the GUI
thread. Results and errors come back as Qt signals on the GUI thread.

Callers must keep a reference to the Worker returned by
run_in_background() until it finishes (e.g. append it to a list on the
owning widget, remove it in the finished/failed callback) -- nothing
else keeps it alive.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class WorkerSignals(QObject):
    finished = Signal(object)  # the callable's return value
    failed = Signal(Exception)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:  # noqa: BLE001 -- surfaced to the caller via `failed`, not swallowed
            self.signals.failed.emit(e)
        else:
            self.signals.finished.emit(result)


def run_in_background(fn, *args, on_finished=None, on_failed=None, **kwargs) -> Worker:
    worker = Worker(fn, *args, **kwargs)
    if on_finished is not None:
        worker.signals.finished.connect(on_finished)
    if on_failed is not None:
        worker.signals.failed.connect(on_failed)
    QThreadPool.globalInstance().start(worker)
    return worker
