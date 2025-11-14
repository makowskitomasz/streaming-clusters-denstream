from fastapi import FastAPI

from controllers.health_controller import health_api
from controllers.stream_controller import router as stream_api
from controllers.nyc_taxi_controller import router as nyc_taxi_api


def create_app() -> FastAPI:
    app = FastAPI(title="Clustering API")
    app.include_router(health_api)
    app.include_router(stream_api)
    app.include_router(nyc_taxi_api)
    return app
