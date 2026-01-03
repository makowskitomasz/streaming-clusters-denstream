"""Shared abstraction for clustering engines used across the streaming pipeline.

Every concrete clusterer (DenStream, HDBSCAN, future models) should implement
this interface so services can remain agnostic to underlying algorithms.
"""

import abc
from typing import Any


class BaseClusterer(abc.ABC):
    """Common contract for clustering adapters."""

    @abc.abstractmethod
    def fit(self, data: Any) -> None:
        """Train/initialize the clusterer on the provided historical data."""

    @abc.abstractmethod
    def update(self, data: Any) -> None:
        """Incrementally update the model with the next data batch."""

    @abc.abstractmethod
    def get_clusters(self) -> Any:
        """Return the current cluster assignments/structures."""

