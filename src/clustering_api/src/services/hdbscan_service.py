from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from hdbscan import HDBSCAN
from loguru import logger

from clustering_api.src.services.metrics_service import MetricsService, metrics_service
from clustering_api.src.utils.latency import measure_latency


@dataclass(frozen=True, slots=True)
class HdbscanBatchMetadata:
    """Metadata describing a processed batch."""

    batch_id: str | None
    timestamp: float
    n_samples: int


@dataclass(frozen=True, slots=True)
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
        metrics: MetricsService | None = None,
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
        self._history: deque[HdbscanBatchMetadata] = deque(maxlen=history_size)
        self._metrics = metrics or metrics_service

    def cluster_batch(
        self, features: np.ndarray, batch_id: str | None = None,
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
            msg = f"features must be a 2D array-like structure, got {data.ndim}D"
            raise ValueError(msg)
        n_samples = int(data.shape[0])
        self._append_history(batch_id=batch_id, n_samples=n_samples)
        if n_samples == 0:
            self._log_batch_stats(
                n_samples=0,
                number_of_clusters=0,
                noise_ratio=0.0,
                batch_id=batch_id,
                latency_ms=0.0,
            )
            return HdbscanBatchResult(
                labels=[],
                number_of_clusters=0,
                noise_ratio=0.0,
                silhouette_score=None,
                cluster_size_summary=None,
                batch_id=batch_id,
                n_samples=0,
            )

        with measure_latency() as timer:
            clusterer = self._build_clusterer()
            labels = self._fit_predict(clusterer, data)
            metrics_record = self._metrics.evaluate(
                data,
                labels,
                model_name="hdbscan",
                batch_id=batch_id,
            )
        self._log_batch_stats(
            n_samples=metrics_record.n_samples,
            number_of_clusters=metrics_record.number_of_clusters,
            noise_ratio=metrics_record.noise_ratio,
            batch_id=metrics_record.batch_id,
            latency_ms=timer.ms,
        )
        return HdbscanBatchResult(
            labels=labels.tolist(),
            number_of_clusters=metrics_record.number_of_clusters,
            noise_ratio=metrics_record.noise_ratio,
            silhouette_score=metrics_record.silhouette_score,
            cluster_size_summary=self._summarize_cluster_sizes(labels),
            batch_id=batch_id,
            n_samples=n_samples,
        )

    def get_history(self) -> tuple[HdbscanBatchMetadata, ...]:
        """Return a read-only snapshot of recent batch metadata."""
        return tuple(self._history)

    def _build_clusterer(self) -> HDBSCAN:
        params = {
            key: value for key, value in self._params.items() if value is not None
        }
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
            ),
        )

    def _summarize_cluster_sizes(
        self, labels: np.ndarray,
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
            msg = f"min_cluster_size must be greater than 0, got {min_cluster_size}"
            raise ValueError(msg)
        if min_samples is not None and min_samples <= 0:
            msg = f"min_samples must be greater than 0 when provided, got {min_samples}"
            raise ValueError(msg)
        if not metric:
            msg = "metric must be a non-empty string"
            raise ValueError(msg)
        if cluster_selection_method not in {"eom", "leaf"}:
            msg = (
                "Invalid cluster_selection_method: "
                f"{cluster_selection_method}, must be 'eom' or 'leaf'"
            )
            raise ValueError(msg)
        if history_size <= 0:
            msg = f"history_size must be greater than 0, got {history_size}"
            raise ValueError(msg)
        if random_state is not None and random_state < 0:
            msg = f"random_state must be non-negative when provided, got {random_state}"
            raise ValueError(msg)

    def _log_batch_stats(
        self,
        *,
        n_samples: int,
        number_of_clusters: int,
        noise_ratio: float,
        batch_id: str | None,
        latency_ms: float,
    ) -> None:
        logger.bind(
            event="clustering_batch",
            model_name="hdbscan",
            batch_id=batch_id,
            n_samples=n_samples,
            active_clusters=number_of_clusters,
            avg_density=None,
            noise_ratio=noise_ratio,
            latency_ms=latency_ms,
            timestamp=time.time(),
        ).info("HDBSCAN batch processed")
