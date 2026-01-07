from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter


@dataclass(slots=True)
class Latency:
    """Container for elapsed milliseconds."""

    ms: float = 0.0


@contextmanager
def measure_latency() -> Iterator[Latency]:
    """Measure elapsed time in milliseconds."""
    start = perf_counter()
    latency = Latency()
    try:
        yield latency
    finally:
        latency.ms = (perf_counter() - start) * 1000
