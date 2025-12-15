import uvicorn

from clustering_api.src.app import create_app
from clustering_api.src.config import config

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.app.server_port,
        reload=True,
    )
