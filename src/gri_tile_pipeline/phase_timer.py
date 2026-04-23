"""Lightweight per-phase timing for the predict step.

Used to surface where the ~180 s/tile budget actually goes so we can make
data-driven decisions about where to optimize (GPU inference, preprocessing
vectorization, batching, region co-location, etc.) rather than guessing.

The ``PhaseTimer`` is a context-manager-based collector: overlapping calls
accumulate time against the same phase key, and a ``None`` timer is a no-op
so callers that don't care pay zero overhead.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional


class PhaseTimer:
    """Accumulate wallclock time per named phase."""

    def __init__(self) -> None:
        self._phases: Dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._phases[name] = self._phases.get(name, 0.0) + dt

    def record(self, name: str, seconds: float) -> None:
        """Add *seconds* to *name* without using a context manager."""
        self._phases[name] = self._phases.get(name, 0.0) + float(seconds)

    def as_dict(self) -> Dict[str, float]:
        """Snapshot of accumulated timings, rounded to 4 decimals (100 µs)."""
        return {k: round(v, 4) for k, v in self._phases.items()}


@contextmanager
def timed(timer: Optional[PhaseTimer], name: str) -> Iterator[None]:
    """Convenience wrapper: no-op when *timer* is None."""
    if timer is None:
        yield
        return
    with timer.phase(name):
        yield
