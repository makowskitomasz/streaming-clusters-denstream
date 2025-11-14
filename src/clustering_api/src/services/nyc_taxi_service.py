import pandas as pd
from pathlib import Path
from typing import List, Iterator, Optional
from clustering_api.src.models.data_models import DataPoint


class NycTaxiService:
    """
    Loads NYC Taxi trip data (2016 format) and converts to streaming DataPoints.
    """

    def __init__(self, file_path: str, batch_size: int = 500):
        self.file_path = Path(file_path)
        self.batch_size = batch_size
        self._iterator = None
        self.batch_id = 0

    # -----------------------
    # Internal helpers
    # -----------------------

    def _load_df(self) -> pd.DataFrame:
        """
        Load CSV/Parquet depending on extension.
        Expected columns:
        - tpep_pickup_datetime
        - pickup_longitude
        - pickup_latitude
        """
        if self.file_path.suffix == ".parquet":
            df = pd.read_parquet(self.file_path)
        else:
            df = pd.read_csv(self.file_path)

        df = df[
            [
                "tpep_pickup_datetime",
                "pickup_longitude",
                "pickup_latitude",
                "trip_distance",
                "total_amount",
            ]
        ].dropna()

        # sort by time
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        df = df.sort_values("tpep_pickup_datetime")

        return df

    def _df_to_datapoint(self, row) -> DataPoint:
        # fallback cluster id → spatial grid
        grid_x = int(row["pickup_longitude"] * 100)
        grid_y = int(row["pickup_latitude"] * 100)
        cluster_id = abs(hash((grid_x, grid_y))) % 500

        return DataPoint(
            x=float(row["pickup_longitude"]),
            y=float(row["pickup_latitude"]),
            timestamp=row["tpep_pickup_datetime"].timestamp(),
            cluster_id=cluster_id,
            source="nyc_taxi",
        )

    # -----------------------
    # Streaming iterator
    # -----------------------

    def _ensure_iterator(self):
        if self._iterator is None:
            df = self._load_df()
            self._iterator = df.iterrows()

    def next_batch(self) -> Optional[List[DataPoint]]:
        """
        Returns the next batch of DataPoints.
        If fewer than batch_size remain, return them anyway.
        """
        self._ensure_iterator()

        batch: List[DataPoint] = []

        for _ in range(self.batch_size):
            try:
                _, row = next(self._iterator)
                batch.append(self._df_to_datapoint(row))
            except StopIteration:
                break

        if not batch:
            return None

        self.batch_id += 1
        return batch

    def reset(self):
        """
        Reset batch iterator.
        """
        self._iterator = None
        self.batch_id = 0
