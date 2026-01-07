"""Shared abstraction for clustering engines used across the streaming pipeline.

Every concrete clusterer (DenStream, HDBSCAN, future models) should implement
this interface so services can remain agnostic to underlying algorithms.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


class BaseClusterer(abc.ABC):
    """Common contract for clustering adapters."""

    @abc.abstractmethod
    def fit(self, data: Iterable[Any]) -> None:
        """Train/initialize the clusterer on the provided historical data."""

    @abc.abstractmethod
    def update(self, data: Iterable[Any]) -> None:
        """Incrementally update the model with the next data batch."""

    @abc.abstractmethod
    def get_clusters(self) -> dict[str, list[object]]:
        """Return the current cluster assignments/structures."""
