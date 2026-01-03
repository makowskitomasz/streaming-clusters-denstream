import os

import pandas as pd

from clustering_api.src.services.stream_service import StreamService


def test_generate_batch_structure():
    service = StreamService(n_clusters=2, points_per_cluster=10)
    data = service.generate_batch()
    assert isinstance(data, list)
    assert len(data) > 0
    sample = data[0]
    assert sample.source == "synthetic"
    assert sample.batch_id == service.batch_id
    assert sample.noise in {True, False}


def test_save_batch_creates_file(tmp_path):
    service = StreamService(output_dir=tmp_path)
    file_path = service.save_batch()
    assert os.path.exists(file_path)
    df = pd.read_json(file_path)
    assert len(df) > 0
    assert "x" in df.columns


def test_configure_updates_parameters():
    service = StreamService()
    old_config = service.get_state()
    service.configure(n_clusters=5, points_per_cluster=20, noise_ratio=0.1, drift=0.2)
    new_config = service.get_state()

    assert new_config["n_clusters"] == 5
    assert new_config["points_per_cluster"] == 20
    assert new_config["noise_ratio"] == 0.1
    assert new_config["drift"] == 0.2
    assert len(new_config["centroids"]) == 5
    assert old_config["centroids"] != new_config["centroids"]


def test_reset_stream_resets_batch_and_centroids():
    service = StreamService()
    service.generate_batch()
    assert service.batch_id > 0

    old_centroids = service.get_state()["centroids"]
    service.reset_stream()
    new_state = service.get_state()

    assert new_state["batch_id"] == 0
    assert old_centroids != new_state["centroids"]
