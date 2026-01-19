import csv
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from clustering_api.src.models.data_models import (
    ClusterPoint,
    DataPoint,
    map_datapoint_to_clusterpoint,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class NycTaxiService:
    """Load NYC Taxi trip data (2016) and stream it in fixed batches."""

    MIN_BOUNDS_SAMPLES = 2
    NYC_LONGITUDE_RANGE = (-74.3, -73.6)
    NYC_LATITUDE_RANGE = (40.45, 40.95)

    def __init__(self, file_path: str, batch_size: int = 500) -> None:
        """Initialize NYC Taxi stream with a data source and batch size."""
        self._file_path = Path(file_path)
        self._batch_size = batch_size
        self._iterator: Iterator[tuple[int, pd.Series]] | None = None
        self._pending_row: tuple[int, pd.Series] | None = None
        self.batch_id = 0
        self._cached_bounds: tuple[float, float, float, float] | None = None

    @property
    def batch_size(self) -> int:
        """Return current batch size."""
        return self._batch_size

    @property
    def file_path(self) -> Path:
        """Return the current data source path."""
        return self._file_path

    def load_df(self) -> pd.DataFrame:
        """Load CSV/Parquet file, keep relevant columns, and order chronologically."""
        match self._file_path.suffix:
            case ".parquet":
                df = pd.read_parquet(self._file_path)
            case _:
                df = pd.read_csv(self._file_path)

        df = df[
            [
                "tpep_pickup_datetime",
                "pickup_longitude",
                "pickup_latitude",
            ]
        ]
        df["pickup_longitude"] = pd.to_numeric(df["pickup_longitude"], errors="coerce")
        df["pickup_latitude"] = pd.to_numeric(df["pickup_latitude"], errors="coerce")
        df = df.dropna()
        lon_min, lon_max = self.NYC_LONGITUDE_RANGE
        lat_min, lat_max = self.NYC_LATITUDE_RANGE
        df = df[
            (df["pickup_longitude"] != 0)
            & (df["pickup_latitude"] != 0)
            & (df["pickup_longitude"].between(lon_min, lon_max))
            & (df["pickup_latitude"].between(lat_min, lat_max))
        ]

        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        return df.sort_values("tpep_pickup_datetime")

    def _df_to_datapoint(self, row: pd.Series) -> DataPoint:
        """Convert a raw row into a DataPoint."""
        return self._row_to_datapoint(row)

    def _df_to_clusterpoint(self, row: pd.Series) -> ClusterPoint:
        """Convert a raw row into a ClusterPoint."""
        return map_datapoint_to_clusterpoint(
            self._row_to_datapoint(row),
        )

    def _ensure_iterator(self) -> None:
        """Create a row iterator if not already initialized."""
        if self._iterator is None:
            df = self.load_df()
            self._iterator = df.iterrows()
            self._pending_row = None

    def next_batch(self) -> list[DataPoint] | None:
        """Return the next batch of data points or ``None`` if stream is exhausted."""
        self._ensure_iterator()

        rows = list(islice(self._iterator, self._batch_size))
        if not rows:
            return None

        self.batch_id += 1
        return [self._df_to_datapoint(row) for _, row in rows]

    def next_batch_cluster_points(self) -> list[ClusterPoint] | None:
        """Return the next batch of cluster points or ``None`` if exhausted."""
        self._ensure_iterator()

        rows = list(islice(self._iterator, self._batch_size))
        if not rows:
            return None

        self.batch_id += 1
        return [self._df_to_clusterpoint(row) for _, row in rows]

    def next_second(self) -> list[DataPoint] | None:
        """Return all points for the next second in the stream."""
        self._ensure_iterator()
        first = self._pending_row or next(self._iterator, None)
        self._pending_row = None
        if first is None:
            return None
        _, first_row = first
        target_second = int(pd.to_datetime(first_row["tpep_pickup_datetime"]).timestamp())
        rows = [first_row]
        for next_item in self._iterator:
            _, row = next_item
            current_second = int(pd.to_datetime(row["tpep_pickup_datetime"]).timestamp())
            if current_second != target_second:
                self._pending_row = next_item
                break
            rows.append(row)
        self.batch_id += 1
        return [self._df_to_datapoint(row) for row in rows]

    def next_second_cluster_points(self) -> list[ClusterPoint] | None:
        """Return all cluster points for the next second in the stream."""
        batch = self.next_second()
        if batch is None:
            return None
        return [map_datapoint_to_clusterpoint(point) for point in batch]

    def reset(self) -> None:
        """Reset internal iterator and batch counter."""
        self._iterator = None
        self._pending_row = None
        self.batch_id = 0
        self._cached_bounds = None

    def configure(self, file_path: str | None = None, batch_size: int | None = None) -> None:
        """Update source file path or batch size and reset iterator."""
        if file_path is not None:
            self._file_path = Path(file_path)
        if batch_size is not None and batch_size > 0:
            self._batch_size = batch_size
        self.reset()

    def bounds(self) -> tuple[float, float, float, float] | None:
        """Compute approximate bounds from a sample of the input file."""
        if self._cached_bounds is not None:
            return self._cached_bounds
        if not self._file_path.exists():
            return None
        count = 0
        mean_lon = 0.0
        mean_lat = 0.0
        m2_lon = 0.0
        m2_lat = 0.0
        found = False
        try:
            with self._file_path.open() as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    try:
                        lon = float(row["pickup_longitude"])
                        lat = float(row["pickup_latitude"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if lon == 0 or lat == 0:
                        continue
                    count += 1
                    delta_lon = lon - mean_lon
                    mean_lon += delta_lon / count
                    m2_lon += delta_lon * (lon - mean_lon)
                    delta_lat = lat - mean_lat
                    mean_lat += delta_lat / count
                    m2_lat += delta_lat * (lat - mean_lat)
                    found = True
        except FileNotFoundError:
            return None
        if not found:
            return None
        if count < self.MIN_BOUNDS_SAMPLES:
            return None
        std_lon = (m2_lon / (count - 1)) ** 0.5
        std_lat = (m2_lat / (count - 1)) ** 0.5
        min_lon = mean_lon - 3 * std_lon
        max_lon = mean_lon + 3 * std_lon
        min_lat = mean_lat - 3 * std_lat
        max_lat = mean_lat + 3 * std_lat
        self._cached_bounds = (min_lon, max_lon, min_lat, max_lat)
        return self._cached_bounds

    def _row_to_datapoint(self, row: pd.Series) -> DataPoint:
        """Convert a raw row into a DataPoint with a grid-based cluster id."""
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
