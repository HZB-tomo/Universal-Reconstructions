"""
Tests for orchestration/ram_check.py's estimate_chunk_rows(): the
chunk-sizing logic PipelineRun._run_streaming() falls back to when
ProjectConfig.stream_chunk_rows isn't given explicitly. RAM/CPU count
are mocked so these are deterministic regardless of the machine
running them.
"""
from orchestration import ram_check


class _FakeVirtualMemory:
    def __init__(self, available: int):
        self.available = available


def test_estimate_chunk_rows_returns_zero_when_there_are_no_rows_to_give_it():
    assert ram_check.estimate_chunk_rows(width=300, max_rows=0) == 0


def test_estimate_chunk_rows_uses_cpu_count_when_ram_and_rows_both_allow_it(monkeypatch):
    monkeypatch.setattr(ram_check.psutil, "virtual_memory", lambda: _FakeVirtualMemory(available=100 * 1024**3))
    monkeypatch.setattr(ram_check.os, "cpu_count", lambda: 8)
    assert ram_check.estimate_chunk_rows(width=300, max_rows=10_000) == 8


def test_estimate_chunk_rows_is_capped_by_available_ram(monkeypatch):
    # Just enough available RAM for ~2 rows at this width once the
    # safety margin is applied -- see estimate_volume_bytes()/SAFETY_FACTOR.
    per_row_bytes = ram_check.estimate_volume_bytes(1, 300) * ram_check.SAFETY_FACTOR
    monkeypatch.setattr(ram_check.psutil, "virtual_memory", lambda: _FakeVirtualMemory(available=int(per_row_bytes * 2.5)))
    monkeypatch.setattr(ram_check.os, "cpu_count", lambda: 8)
    assert ram_check.estimate_chunk_rows(width=300, max_rows=10_000) == 2


def test_estimate_chunk_rows_is_capped_by_the_remaining_row_count(monkeypatch):
    monkeypatch.setattr(ram_check.psutil, "virtual_memory", lambda: _FakeVirtualMemory(available=100 * 1024**3))
    monkeypatch.setattr(ram_check.os, "cpu_count", lambda: 8)
    assert ram_check.estimate_chunk_rows(width=300, max_rows=3) == 3


def test_estimate_chunk_rows_never_returns_less_than_one_row(monkeypatch):
    monkeypatch.setattr(ram_check.psutil, "virtual_memory", lambda: _FakeVirtualMemory(available=1))
    monkeypatch.setattr(ram_check.os, "cpu_count", lambda: 8)
    assert ram_check.estimate_chunk_rows(width=300, max_rows=10_000) == 1
