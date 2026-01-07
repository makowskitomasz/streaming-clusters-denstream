import pandas as pd

from clustering_api.src.services.nyc_taxi_service import NycTaxiService

# --------------------------
# Helper: create small CSV
# --------------------------

def create_sample_csv(tmp_path):
    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                "2016-01-01 00:00:00",
                "2016-01-01 00:01:00",
                "2016-01-01 00:02:00",
                "2016-01-01 00:03:00",
            ],
            "pickup_longitude": [-73.99, -73.98, -73.97, -73.96],
            "pickup_latitude": [40.73, 40.74, 40.75, 40.76],
            "trip_distance": [1.1, 2.2, 3.3, 4.4],
            "total_amount": [8.80, 12.50, 16.30, 20.10],
        },
    )
    test_file = tmp_path / "nyc_sample.csv"
    df.to_csv(test_file, index=False)
    return str(test_file)


# --------------------------
# TESTS
# --------------------------

def test_load_dataframe(tmp_path):
    file_path = create_sample_csv(tmp_path)
    service = NycTaxiService(file_path=file_path, batch_size=2)

    df = service.load_df()
    assert len(df) == 4
    assert "pickup_longitude" in df.columns
    assert "pickup_latitude" in df.columns
    assert df["tpep_pickup_datetime"].dtype == "datetime64[ns]"


def test_next_batch_returns_datapoints(tmp_path):
    file_path = create_sample_csv(tmp_path)
    service = NycTaxiService(file_path=file_path, batch_size=2)

    batch = service.next_batch()

    assert batch is not None
    assert len(batch) == 2

    # Validate DataPoint structure
    point = batch[0]
    assert hasattr(point, "x")
    assert hasattr(point, "y")
    assert hasattr(point, "timestamp")
    assert hasattr(point, "cluster_id")
    assert point.source == "nyc_taxi"
    assert point.batch_id == service.batch_id
    assert point.noise is None


def test_next_batch_order(tmp_path):
    file_path = create_sample_csv(tmp_path)
    service = NycTaxiService(file_path=file_path, batch_size=2)

    b1 = service.next_batch()
    b2 = service.next_batch()

    assert b1 is not None and b2 is not None
    assert b1[0].timestamp < b2[0].timestamp  # chronological order


def test_next_batch_end_of_data(tmp_path):
    file_path = create_sample_csv(tmp_path)
    service = NycTaxiService(file_path=file_path, batch_size=3)

    b1 = service.next_batch()
    b2 = service.next_batch()  # should be smaller
    b3 = service.next_batch()  # should be None

    assert b1 is not None
    assert b2 is not None
    assert b3 is None  # we reached the end


def test_reset_stream(tmp_path):
    file_path = create_sample_csv(tmp_path)
    service = NycTaxiService(file_path=file_path, batch_size=2)

    service.next_batch()
    assert service.batch_id == 1

    service.reset()
    assert service.batch_id == 0

    batch = service.next_batch()
    assert batch is not None
    assert len(batch) == 2  # should start from the beginning again


def test_cluster_id_grid_hashing(tmp_path):
    file_path = create_sample_csv(tmp_path)
    service = NycTaxiService(file_path=file_path, batch_size=1)

    batch = service.next_batch()

    point = batch[0]
    assert isinstance(point.cluster_id, int)
    assert point.cluster_id >= 0
    assert point.batch_id == service.batch_id
