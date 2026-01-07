from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Query

from clustering_api.src.utils.logging_utils import get_recent_logs

router = APIRouter(prefix="/v1/logs", tags=["Logs"])


@dataclass(frozen=True, slots=True)
class LogsResponse:
    logs: list[dict[str, object]]

    def to_dict(self) -> dict[str, list[dict[str, object]]]:
        return {"logs": self.logs}


@router.get("/recent", summary="Fetch recent backend logs")
def recent_logs(
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> dict[str, list[dict[str, object]]]:
    response = LogsResponse(logs=get_recent_logs(limit))
    return response.to_dict()
