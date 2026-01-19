from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

NYC_LAT0 = 40.75
_M_PER_DEG_LAT = 111_320.0
_M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(NYC_LAT0))


@dataclass(slots=True)
class Latency:
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


def _lonlat_to_m(lon: float, lat: float) -> tuple[float, float]:
    return lon * _M_PER_DEG_LON, lat * _M_PER_DEG_LAT
