from fastapi import APIRouter

health_api = APIRouter(prefix="/v1/health", tags=["Health"])

@health_api.get("")
def health():
    return {"status": "ok"}
