from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score


@dataclass(frozen=True)
class HdbscanBatchMetadata:
    """Metadata describing a processed batch."""

    batch_id: str | None
    timestamp: float
    n_samples: int


@dataclass(frozen=True)
class HdbscanBatchResult:
    """Result of running HDBSCAN on a single batch.

    Attributes:
        labels: Cluster labels for each sample; noise is labeled -1.
        number_of_clusters: Count of clusters excluding noise (-1).
        noise_ratio: Fraction of samples labeled as noise.
        silhouette_score: Silhouette score for non-noise clusters, or None when
            fewer than two clusters are present or there are not enough samples.
        cluster_size_summary: Min/mean/max cluster sizes for non-noise clusters,
            or None when no clusters exist.
    """

    labels: list[int]
    number_of_clusters: int
    noise_ratio: float
    silhouette_score: float | None
    cluster_size_summary: dict[str, float] | None
    batch_id: str | None
    n_samples: int


class HdbscanService:
    """Offline baseline clustering service using HDBSCAN."""

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
        random_state: int | None = None,
        history_size: int = 50,
    ) -> None:
        self._validate_params(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
            history_size=history_size,
            random_state=random_state,
        )
        self._params = {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "metric": metric,
            "cluster_selection_method": cluster_selection_method,
        }
        self._random_state = random_state
        self._history_size = history_size
        self._history: list[HdbscanBatchMetadata] = []

    def cluster_batch(
        self, features: np.ndarray, batch_id: str | None = None
    ) -> HdbscanBatchResult:
        """Cluster a batch of features with HDBSCAN.

        Args:
            features: 2D array-like of shape (n_samples, n_features).
            batch_id: Optional identifier for batch tracking.

        Returns:
            HdbscanBatchResult with labels and evaluation metrics.
        """
        data = np.asarray(features)
        if data.ndim != 2:
            raise ValueError("features must be a 2D array-like structure")
        n_samples = int(data.shape[0])
        self._append_history(batch_id=batch_id, n_samples=n_samples)
        if n_samples == 0:
            return HdbscanBatchResult(
                labels=[],
                number_of_clusters=0,
                noise_ratio=0.0,
                silhouette_score=None,
                cluster_size_summary=None,
                batch_id=batch_id,
                n_samples=0,
            )

        clusterer = self._build_clusterer()
        labels = self._fit_predict(clusterer, data)
        metrics = self._compute_metrics(data, labels)
        return HdbscanBatchResult(
            labels=labels.tolist(),
            number_of_clusters=metrics["number_of_clusters"],
            noise_ratio=metrics["noise_ratio"],
            silhouette_score=metrics["silhouette_score"],
            cluster_size_summary=metrics["cluster_size_summary"],
            batch_id=batch_id,
            n_samples=n_samples,
        )

    def get_history(self) -> tuple[HdbscanBatchMetadata, ...]:
        """Return a read-only snapshot of recent batch metadata."""
        return tuple(self._history)

    def _build_clusterer(self) -> HDBSCAN:
        params = {key: value for key, value in self._params.items() if value is not None}
        return HDBSCAN(**params)

    def _fit_predict(self, clusterer: HDBSCAN, data: np.ndarray) -> np.ndarray:
        if self._random_state is None:
            return clusterer.fit_predict(data)
        rng_state = np.random.get_state()
        np.random.seed(self._random_state)
        try:
            return clusterer.fit_predict(data)
        finally:
            np.random.set_state(rng_state)

    def _append_history(self, batch_id: str | None, n_samples: int) -> None:
        self._history.append(
            HdbscanBatchMetadata(
                batch_id=batch_id,
                timestamp=time.time(),
                n_samples=n_samples,
            )
        )
        if len(self._history) > self._history_size:
            excess = len(self._history) - self._history_size
            self._history = self._history[excess:]

    def _compute_metrics(
        self, data: np.ndarray, labels: np.ndarray
    ) -> dict[str, float | int | None | dict[str, float]]:
        n_samples = int(labels.size)
        if n_samples == 0:
            return {
                "number_of_clusters": 0,
                "noise_ratio": 0.0,
                "silhouette_score": None,
                "cluster_size_summary": None,
            }

        noise_count = int(np.sum(labels == -1))
        noise_ratio = noise_count / n_samples if n_samples else 0.0
        unique_labels = {int(label) for label in set(labels.tolist()) if label != -1}
        number_of_clusters = len(unique_labels)

        cluster_size_summary = self._summarize_cluster_sizes(labels)
        silhouette = self._safe_silhouette_score(data, labels, number_of_clusters)

        return {
            "number_of_clusters": number_of_clusters,
            "noise_ratio": noise_ratio,
            "silhouette_score": silhouette,
            "cluster_size_summary": cluster_size_summary,
        }

    def _summarize_cluster_sizes(
        self, labels: np.ndarray
    ) -> dict[str, float] | None:
        cluster_labels = labels[labels != -1]
        if cluster_labels.size == 0:
            return None
        counts = np.bincount(cluster_labels.astype(int))
        counts = counts[counts > 0]
        if counts.size == 0:
            return None
        return {
            "min": float(np.min(counts)),
            "mean": float(np.mean(counts)),
            "max": float(np.max(counts)),
        }

    def _safe_silhouette_score(
        self, data: np.ndarray, labels: np.ndarray, number_of_clusters: int
    ) -> float | None:
        if number_of_clusters < 2:
            return None
        mask = labels != -1
        if np.sum(mask) < 2:
            return None
        clustered_labels = labels[mask]
        if len(set(clustered_labels.tolist())) < 2:
            return None
        return float(silhouette_score(data[mask], clustered_labels))

    def _validate_params(
        self,
        min_cluster_size: int,
        min_samples: int | None,
        metric: str,
        cluster_selection_method: str,
        history_size: int,
        random_state: int | None,
    ) -> None:
        if min_cluster_size <= 0:
            raise ValueError("min_cluster_size must be greater than 0")
        if min_samples is not None and min_samples <= 0:
            raise ValueError("min_samples must be greater than 0 when provided")
        if not metric:
            raise ValueError("metric must be a non-empty string")
        if cluster_selection_method not in {"eom", "leaf"}:
            raise ValueError(
                "cluster_selection_method must be 'eom' or 'leaf'"
            )
        if history_size <= 0:
            raise ValueError("history_size must be greater than 0")
        if random_state is not None and random_state < 0:
            raise ValueError("random_state must be non-negative when provided")
