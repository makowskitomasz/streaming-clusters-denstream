from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from loguru import logger
from sklearn.metrics import silhouette_score


@dataclass(frozen=True, slots=True)
class MetricsRecord:
    """Snapshot of clustering evaluation metrics."""

    timestamp: float
    model_name: str
    batch_id: str | None
    n_samples: int
    number_of_clusters: int
    noise_ratio: float
    silhouette_score: float | None


class MetricsService:
    """Compute and store clustering metrics for monitoring."""

    def __init__(self, history_size: int = 100) -> None:
        if history_size <= 0:
            msg = f"history_size must be greater than 0, got {history_size}"
            raise ValueError(msg)
        self._history_size = history_size
        self._history: dict[str, deque[MetricsRecord]] = {}

    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        model_name: str,
        batch_id: str | None = None,
    ) -> MetricsRecord:
        """Compute metrics for a batch and store the result."""
        if not model_name:
            msg = "model_name must be a non-empty string"
            raise ValueError(msg)
        data = np.asarray(features)
        label_array = np.asarray(labels)
        if data.ndim != 2:
            msg = f"features must be a 2D array-like structure, got {data.ndim}D"
            raise ValueError(msg)
        if label_array.ndim != 1:
            msg = f"labels must be a 1D array-like structure, got {label_array.ndim}D"
            raise ValueError(msg)
        n_samples = int(data.shape[0])
        if n_samples != int(label_array.size):
            msg = (
                "features and labels must have matching lengths, "
                f"got {n_samples} and {label_array.size}"
            )
            raise ValueError(msg)

        number_of_clusters = self._count_clusters(label_array)
        noise_ratio = self._compute_noise_ratio(label_array, n_samples)
        silhouette = self._safe_silhouette_score(data, label_array, number_of_clusters)

        record = MetricsRecord(
            timestamp=time.time(),
            model_name=model_name,
            batch_id=batch_id,
            n_samples=n_samples,
            number_of_clusters=number_of_clusters,
            noise_ratio=noise_ratio,
            silhouette_score=silhouette,
        )
        self._store(record)
        self._log(record)
        return record

    def get_latest(
        self, model_name: str | None = None,
    ) -> MetricsRecord | None | dict[str, MetricsRecord]:
        """Return the latest metrics record for a model or for all models."""
        if model_name is None:
            return {
                name: records[-1] for name, records in self._history.items() if records
            }
        records = self._history.get(model_name)
        return records[-1] if records else None

    def get_history(self, model_name: str) -> tuple[MetricsRecord, ...]:
        """Return a read-only copy of stored metrics for a model."""
        records = self._history.get(model_name)
        return tuple(records) if records else ()

    def _store(self, record: MetricsRecord) -> None:
        records = self._history.setdefault(
            record.model_name,
            deque(maxlen=self._history_size),
        )
        records.append(record)

    def _count_clusters(self, labels: np.ndarray) -> int:
        return len({int(label) for label in labels.tolist() if int(label) != -1})

    def _compute_noise_ratio(self, labels: np.ndarray, n_samples: int) -> float:
        if n_samples == 0:
            return 0.0
        noise_count = int(np.sum(labels == -1))
        return noise_count / n_samples

    def _safe_silhouette_score(
        self, data: np.ndarray, labels: np.ndarray, number_of_clusters: int,
    ) -> float | None:
        if data.shape[0] < 2 or number_of_clusters < 2:
            return None
        mask = labels != -1
        if int(np.sum(mask)) < 2:
            return None
        clustered_labels = labels[mask]
        if len(set(clustered_labels.tolist())) < 2:
            return None
        return float(silhouette_score(data[mask], clustered_labels))

    def _log(self, record: MetricsRecord) -> None:
        if record.n_samples == 0 or record.number_of_clusters == 0:
            logger.warning(
                "metrics computed | model={model} batch={batch} n_samples={n} "
                "n_clusters={clusters} noise_ratio={noise:.3f} silhouette={silhouette}",
                model=record.model_name,
                batch=record.batch_id,
                n=record.n_samples,
                clusters=record.number_of_clusters,
                noise=record.noise_ratio,
                silhouette=record.silhouette_score,
            )
            return
        logger.info(
            "metrics computed | model={model} batch={batch} n_samples={n} "
            "n_clusters={clusters} noise_ratio={noise:.3f} silhouette={silhouette}",
            model=record.model_name,
            batch=record.batch_id,
            n=record.n_samples,
            clusters=record.number_of_clusters,
            noise=record.noise_ratio,
            silhouette=record.silhouette_score,
        )


metrics_service = MetricsService()
