from fastapi import APIRouter

health_api = APIRouter(prefix="/v1/health", tags=["Health"])


@health_api.get("")
def health() -> dict[str, str]:
    """Return health-check payload for uptime probes."""
    return {"status": "ok"}
