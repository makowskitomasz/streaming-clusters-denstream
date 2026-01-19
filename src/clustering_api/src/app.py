from fastapi import FastAPI

from clustering_api.src.controllers.clustering_controller import (
    router as clustering_api,
)
from clustering_api.src.controllers.health_controller import health_api
from clustering_api.src.controllers.logs_controller import router as logs_api
from clustering_api.src.controllers.metrics_controller import router as metrics_api
from clustering_api.src.controllers.nyc_taxi_controller import router as nyc_taxi_api
from clustering_api.src.controllers.stream_controller import router as stream_api
from clustering_api.src.utils.logging_utils import init_logging


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    init_logging()
    app = FastAPI(title="Clustering API")
    app.include_router(health_api)
    app.include_router(stream_api)
    app.include_router(nyc_taxi_api)
    app.include_router(clustering_api)
    app.include_router(metrics_api)
    app.include_router(logs_api)
    return app
