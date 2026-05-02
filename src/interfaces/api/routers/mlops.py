"""
MLOps Router — Model Registry & Drift Status

Endpoints:
  GET /v1/mlops/status   → Current model version, drift status
  GET /v1/mlops/models   → List of registered models from MLflow
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter

from src.interfaces.api.schemas import MLOpsStatusResponse, ModelRegistryEntry
from src.interfaces.api.dependencies import model

logger = logging.getLogger(__name__)

router = APIRouter()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
MODEL_REGISTRY_NAME = "cardiorisk-lgbm"


@router.get("/status", response_model=MLOpsStatusResponse)
async def get_mlops_status() -> dict:
    """Current model version, drift detection status."""
    return {
        "current_model_version": model.model_version,
        "model_name": MODEL_REGISTRY_NAME,
        "last_drift_check": None,
        "drift_detected": False,
        "drift_scores": None,
        "is_training": False,
    }


@router.get("/models", response_model=list[ModelRegistryEntry])
async def list_registered_models() -> list[dict]:
    """List all registered model versions from MLflow registry."""
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

        entries = []
        for mv in client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'"):
            # Fetch run metrics
            run = client.get_run(mv.run_id)
            metrics = run.data.metrics

            entries.append({
                "version": mv.version,
                "stage": mv.current_stage,
                "auc_roc": metrics.get("auc_roc", 0.0),
                "auprc": metrics.get("auprc"),
                "brier_score": metrics.get("brier_score"),
                "created_at": str(mv.creation_timestamp),
            })

        # Sort by version descending (newest first)
        entries.sort(key=lambda e: int(e["version"]), reverse=True)
        return entries

    except Exception as exc:
        logger.warning("MLflow registry query failed: %s — returning empty list", exc)
        return []
