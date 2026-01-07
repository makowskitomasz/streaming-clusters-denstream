from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter

health_api = APIRouter(prefix="/v1/health", tags=["Health"])


@dataclass(frozen=True, slots=True)
class HealthResponse:
    status: str

    def to_dict(self) -> dict[str, str]:
        return {"status": self.status}


@health_api.get("")
def health() -> dict[str, str]:
    """Return health-check payload for uptime probes."""
    return HealthResponse(status="ok").to_dict()
