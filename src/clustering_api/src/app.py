from fastapi import FastAPI

from controllers.health_controller import health_api

def create_app() -> FastAPI:
    app = FastAPI(title="Clustering API")
    app.include_router(health_api)
    return app
