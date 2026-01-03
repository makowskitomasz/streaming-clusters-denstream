from itertools import islice
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import pandas as pd

from clustering_api.src.models.data_models import DataPoint, ClusterPoint, map_datapoint_to_clusterpoint


class NycTaxiService:
    """Load NYC Taxi trip data (2016) and stream it in fixed batches."""

    def __init__(self, file_path: str, batch_size: int = 500) -> None:
        self._file_path = Path(file_path)
        self._batch_size = batch_size
        self._iterator: Optional[Iterator[Tuple[int, pd.Series]]] = None
        self.batch_id = 0

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _load_df(self) -> pd.DataFrame:
        """Load CSV/Parquet file, keep relevant columns, and order chronologically."""
        if self._file_path.suffix == ".parquet":
            df = pd.read_parquet(self._file_path)
        else:
            df = pd.read_csv(self._file_path)

        df = df[
            [
                "tpep_pickup_datetime",
                "pickup_longitude",
                "pickup_latitude",
                "trip_distance",
                "total_amount",
            ]
        ].dropna()

        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        df = df.sort_values("tpep_pickup_datetime")

        return df

    def _df_to_datapoint(self, row: pd.Series) -> DataPoint:
        grid_x = int(row["pickup_longitude"] * 100)
        grid_y = int(row["pickup_latitude"] * 100)
        cluster_id = abs(hash((grid_x, grid_y))) % 500

        return DataPoint(
            x=float(row["pickup_longitude"]),
            y=float(row["pickup_latitude"]),
            timestamp=row["tpep_pickup_datetime"].timestamp(),
            cluster_id=cluster_id,
            source="nyc_taxi",
            batch_id=self.batch_id,
            noise=None,
        )
        
    def _df_to_clusterpoint(self, row: pd.Series) -> ClusterPoint:
        grid_x = int(row["pickup_longitude"] * 100)
        grid_y = int(row["pickup_latitude"] * 100)
        cluster_id = abs(hash((grid_x, grid_y))) % 500
        
        return map_datapoint_to_clusterpoint(
            DataPoint(
                x=float(row["pickup_longitude"]),
                y=float(row["pickup_latitude"]),
                timestamp=row["tpep_pickup_datetime"].timestamp(),
                cluster_id=cluster_id,
                source="nyc_taxi",
                batch_id=self.batch_id,
                noise=None,
            )
        )

    def _ensure_iterator(self) -> None:
        if self._iterator is None:
            df = self._load_df()
            self._iterator = df.iterrows()

    def next_batch(self) -> Optional[List[DataPoint]]:
        """Return the next batch of data points or ``None`` if stream is exhausted."""
        self._ensure_iterator()

        rows = list(islice(self._iterator, self._batch_size))
        if not rows:
            return None

        self.batch_id += 1
        batch = [self._df_to_datapoint(row) for _, row in rows]
        return batch
    
    def next_batch_cluster_points(self) -> Optional[list[ClusterPoint]]:
        """Return the next batch of cluster points or ``None`` if stream is exhausted."""
        self._ensure_iterator()

        rows = list(islice(self._iterator, self._batch_size))
        if not rows:
            return None

        self.batch_id += 1
        batch = [self._df_to_clusterpoint(row) for _, row in rows]
        return batch        

    def reset(self) -> None:
        """Reset internal iterator and batch counter."""
        self._iterator = None
        self.batch_id = 0
