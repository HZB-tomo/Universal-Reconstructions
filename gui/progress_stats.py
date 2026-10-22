"""
Speed / time-remaining / total-time bookkeeping for the status box. Qt-free.

Feed it every ProgressEvent in order (update()); it follows one run from
its first event to its "Done"/"Failed" event and can say, at any point,
how fast slices are being finished, how long that leaves, and how long the
whole run took.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from orchestration import ProgressEvent

FINISHED_STAGES = ("Done", "Failed")
_WINDOW = 6  # recent (time, slices done) samples the current speed is averaged over


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes} min {secs:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min {secs:02d} s"


@dataclass(frozen=True)
class Stats:
    speed: Optional[float] = None        # slices per second right now; None until measurable
    remaining_s: Optional[float] = None  # time left at that speed
    elapsed_s: float = 0.0               # since the run's first event (until its end, once finished)
    finished: bool = False
    average_speed: Optional[float] = None  # whole run: slices / elapsed, set when finished


class ProgressTracker:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._reset()

    def _reset(self) -> None:
        self._project: Optional[str] = None
        self._start: Optional[float] = None
        self._samples: deque = deque(maxlen=_WINDOW)
        self._last_done = -1
        self._total: Optional[int] = None
        self._finished = False
        self._end: Optional[float] = None

    def update(self, event: ProgressEvent) -> Stats:
        now = self._clock()
        new_run = (
            self._start is None or self._finished
            or event.project_id != self._project
            or (event.done is not None and event.done < self._last_done)
        )
        if new_run:
            self._reset()
            self._start = now
            self._project = event.project_id

        if event.total:
            self._total = event.total
        if event.done is not None:
            self._last_done = event.done
            # Only a change in slices done is a sample: repeated events at the
            # same count (stage changes inside a chunk) would flatten the speed.
            if not self._samples or event.done != self._samples[-1][1]:
                self._samples.append((now, event.done))

        if event.stage in FINISHED_STAGES:
            self._finished = True
            self._end = now  # the run's length is fixed from here on
        return self.stats(now)

    def stats(self, now: Optional[float] = None) -> Stats:
        now = self._clock() if now is None else now
        elapsed = ((self._end if self._end is not None else now) - self._start) if self._start is not None else 0.0
        speed = None
        if len(self._samples) >= 2:
            (t0, d0), (t1, d1) = self._samples[0], self._samples[-1]
            if t1 > t0 and d1 > d0:
                speed = (d1 - d0) / (t1 - t0)
        remaining = None
        if speed and self._total and self._last_done >= 0 and not self._finished:
            remaining = max(self._total - self._last_done, 0) / speed
        average = None
        if self._finished and self._total and elapsed > 0 and self._last_done > 0:
            average = self._last_done / elapsed
        return Stats(speed=speed, remaining_s=remaining, elapsed_s=elapsed, finished=self._finished, average_speed=average)
