"""Health check router."""
from fastapi import APIRouter
from datetime import datetime, timezone

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/")
async def root() -> dict:
    return {
        "service": "CardioRisk XAI API",
        "version": "0.1.0",
        "docs": "/docs",
    }
