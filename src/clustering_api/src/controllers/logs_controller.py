from fastapi import APIRouter, Query

from clustering_api.src.utils.logging_utils import get_recent_logs

router = APIRouter(prefix="/v1/logs", tags=["Logs"])


@router.get("/recent", summary="Fetch recent backend logs")
def recent_logs(limit: int = Query(200, ge=1, le=1000)) -> dict[str, list[dict[str, object]]]:
    return {"logs": get_recent_logs(limit)}
