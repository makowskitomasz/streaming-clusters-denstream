from fastapi import FastAPI

from .controllers.clustering_controller import router as clustering_api
from .controllers.health_controller import health_api
from .controllers.logs_controller import router as logs_api
from .controllers.metrics_controller import router as metrics_api
from .controllers.nyc_taxi_controller import router as nyc_taxi_api
from .controllers.stream_controller import router as stream_api
from .utils.logging_utils import init_logging


def create_app() -> FastAPI:
    init_logging()
    app = FastAPI(title="Clustering API")
    app.include_router(health_api)
    app.include_router(stream_api)
    app.include_router(nyc_taxi_api)
    app.include_router(clustering_api)
    app.include_router(metrics_api)
    app.include_router(logs_api)
    return app
