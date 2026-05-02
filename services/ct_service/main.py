"""
ContinuousTrainingService — MLOps Tier Microservice

Consumes ModelDriftDetected events from RabbitMQ queue: ct.drift.q
Triggers autonomous retrain → simulated shadow comparison → promotion.

Pipeline:
  1. Pull ALL current data from PostgreSQL (including drifted records)
  2. Retrain LightGBM with Optuna HPO (reduced: 20 trials for speed)
  3. Evaluate challenger model on held-out test set
  4. Load champion metrics from MLflow registry
  5. Simulated shadow comparison: challenger vs champion on same test set
  6. If challenger AUC-ROC > champion AUC-ROC → promote to Production
  7. Publish ModelRetrained event (consumed by API for hot-swap)

Run: make ct
     python services/ct_service/main.py
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import mlflow
from mlflow.tracking import MlflowClient

from src.infrastructure.messaging.consumer_base import BaseRabbitMQConsumer
from src.infrastructure.messaging.publisher import RabbitMQPublisher
from src.domain.events.cardiovascular_events import ModelRetrained

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | CTService | %(message)s",
)
logger = logging.getLogger(__name__)

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
CT_N_TRIALS = int(os.getenv("CT_N_TRIALS", "20"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
MODEL_NAME = "cardiorisk-lgbm"


def _get_champion_auc() -> tuple[str, float]:
    """
    Query MLflow registry for the current Production model's AUC-ROC.
    Returns (version_string, auc_roc).
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Get the latest Production version
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not versions:
        logger.warning("No Production model found in registry — first training?")
        return "0", 0.0

    prod_version = versions[0]
    run = client.get_run(prod_version.run_id)
    auc_roc = run.data.metrics.get("auc_roc", 0.0)
    return prod_version.version, auc_roc


class CTConsumer(BaseRabbitMQConsumer):
    """
    Consumes ModelDriftDetected events and triggers full retrain cycle.
    """

    def __init__(self, publisher: RabbitMQPublisher) -> None:
        super().__init__(
            rabbitmq_url=RABBITMQ_URL,
            queue_name="ct.drift.q",
            routing_key="model.drift.detected",
            prefetch_count=1,   # Process one retrain at a time
        )
        self._publisher = publisher
        self._is_training = False

    async def process_message(self, body: dict[str, Any]) -> None:
        if self._is_training:
            logger.warning("Retrain already in progress — skipping duplicate trigger")
            return

        self._is_training = True
        try:
            await self._run_ct_cycle(body)
        finally:
            self._is_training = False

    async def _run_ct_cycle(self, drift_event: dict[str, Any]) -> None:
        """Execute the full CT cycle: retrain → compare → promote → publish."""
        current_version = drift_event.get("current_model_version", "unknown")
        drifted_features = drift_event.get("drifted_features", {})

        logger.info(
            "=== CT Cycle triggered — drifted features: %s ===",
            list(drifted_features.keys()),
        )

        # 1. Get champion metrics
        champion_version, champion_auc = _get_champion_auc()
        logger.info(
            "Champion: version=%s, AUC-ROC=%.4f",
            champion_version, champion_auc,
        )

        # 2. Retrain challenger (run in thread pool to avoid blocking event loop)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._train_challenger)

        challenger_auc = result["metrics"]["auc_roc"]
        challenger_version = result["model_version"]
        logger.info(
            "Challenger: version=%s, AUC-ROC=%.4f",
            challenger_version, challenger_auc,
        )

        # 3. Simulated shadow comparison
        promoted = challenger_auc > champion_auc
        if promoted:
            logger.info(
                "PROMOTED: challenger v%s (%.4f) > champion v%s (%.4f)",
                challenger_version, challenger_auc,
                champion_version, champion_auc,
            )
        else:
            logger.info(
                "NOT PROMOTED: challenger v%s (%.4f) <= champion v%s (%.4f)",
                challenger_version, challenger_auc,
                champion_version, champion_auc,
            )
            # Demote the challenger back from Production
            # (run_training_pipeline promotes by default; undo if not better)
            try:
                client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=challenger_version,
                    stage="Archived",
                )
                # Re-promote the original champion
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=champion_version,
                    stage="Production",
                    archive_existing_versions=False,
                )
            except Exception as exc:
                logger.error("Failed to revert promotion: %s", exc)

        # 4. Publish ModelRetrained event
        event = ModelRetrained(
            new_model_version=str(challenger_version),
            old_model_version=str(champion_version),
            auc_roc_new=challenger_auc,
            auc_roc_old=champion_auc,
            promoted=promoted,
        )
        await self._publisher.publish(event)
        logger.info(
            "Published ModelRetrained event: promoted=%s, event_id=%s",
            promoted, event.event_id,
        )

    def _train_challenger(self) -> dict:
        """
        Synchronous training — runs in a thread pool.
        Reuses the existing training pipeline with reduced trials.
        """
        import importlib
        # Module name starts with '04_' which isn't directly importable
        train_module = importlib.import_module("ml.pipelines.04_train_lgbm")

        return train_module.run_training_pipeline(
            n_trials=CT_N_TRIALS,
            register_as_production=True,  # Will be reverted if not promoted
        )


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    logger.info("ContinuousTrainingService starting...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    publisher = RabbitMQPublisher()
    await publisher.connect()

    consumer = CTConsumer(publisher=publisher)

    try:
        await consumer.start_consuming()
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    finally:
        await consumer.stop()
        await publisher.close()
        logger.info("ContinuousTrainingService stopped cleanly")


if __name__ == "__main__":
    asyncio.run(main())
