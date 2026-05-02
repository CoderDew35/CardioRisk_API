"""
DriftDetectionService — MLOps Tier Microservice

Consumes PatientTelemetryReceived events from RabbitMQ queue: drift.telemetry.q
Accumulates a sliding window of feature vectors and runs statistical drift tests
(Kolmogorov-Smirnov per feature + Population Stability Index).

When drift is detected, publishes ModelDriftDetected event to trigger the
ContinuousTraining pipeline.

Run: make drift
     python services/drift_service/main.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import numpy as np
import pandas as pd
from scipy import stats

from src.infrastructure.messaging.consumer_base import BaseRabbitMQConsumer
from src.infrastructure.messaging.publisher import RabbitMQPublisher
from src.domain.events.cardiovascular_events import ModelDriftDetected

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | DriftService | %(message)s",
)
logger = logging.getLogger(__name__)

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
DRIFT_WINDOW_SIZE = int(os.getenv("DRIFT_WINDOW_SIZE", "500"))
KS_P_THRESHOLD = float(os.getenv("KS_P_THRESHOLD", "0.01"))
KS_MIN_FEATURES = int(os.getenv("KS_MIN_FEATURES", "3"))
PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.2"))

FEATURE_NAMES = [
    "age_years", "gender", "height_cm", "weight_kg",
    "ap_hi", "ap_lo", "cholesterol", "glucose",
    "is_smoker", "drinks_alcohol", "is_physically_active",
    "bmi", "pulse_pressure", "mean_arterial_pressure", "bp_category_encoded",
]


# ── PSI Calculation ──────────────────────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index between two distributions.
    PSI < 0.1  → no shift
    PSI 0.1–0.2 → moderate shift
    PSI > 0.2  → significant shift
    """
    eps = 1e-6
    breakpoints = np.linspace(
        min(expected.min(), actual.min()) - eps,
        max(expected.max(), actual.max()) + eps,
        n_bins + 1,
    )
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected) + eps
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual) + eps
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


# ── Reference Distribution ──────────────────────────────────────────────────

def load_reference_distribution() -> pd.DataFrame:
    """Load the first 20K training records from PostgreSQL as reference."""
    import sqlalchemy

    db_url = os.getenv("DATABASE_URL", "").replace("postgresql+asyncpg://", "postgresql://")
    engine = sqlalchemy.create_engine(db_url)
    query = """
        SELECT
            age_years, gender, height_cm, weight_kg,
            ap_hi, ap_lo, cholesterol, glucose,
            is_smoker::int, drinks_alcohol::int, is_physically_active::int,
            bmi, bp_category_encoded
        FROM patient_cardiovascular_records
        WHERE data_source = 'batch_csv'
        ORDER BY id
        LIMIT 20000
    """
    df = pd.read_sql(query, engine)
    engine.dispose()

    # Derive engineered features
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["mean_arterial_pressure"] = df["ap_lo"] + df["pulse_pressure"] / 3.0
    logger.info("Reference distribution loaded: %d records", len(df))
    return df


# ── Drift Consumer ───────────────────────────────────────────────────────────

class DriftConsumer(BaseRabbitMQConsumer):
    """
    Accumulates feature vectors in a sliding window.
    Every DRIFT_WINDOW_SIZE records, runs KS + PSI tests against reference.
    """

    def __init__(self, publisher: RabbitMQPublisher, reference_df: pd.DataFrame) -> None:
        super().__init__(
            rabbitmq_url=RABBITMQ_URL,
            queue_name="drift.telemetry.q",
            routing_key="patient.telemetry.raw",
            prefetch_count=50,
        )
        self._publisher = publisher
        self._reference = reference_df
        self._window: deque[dict[str, float]] = deque(maxlen=DRIFT_WINDOW_SIZE)
        self._records_since_check = 0
        self._model_version = os.getenv("MODEL_VERSION", "v1")

    async def process_message(self, body: dict[str, Any]) -> None:
        """Extract feature vector from telemetry event and accumulate."""
        raw = body.get("raw_payload", {})
        if not raw:
            return

        # Build feature dict from raw payload (same mapping as ingest use case)
        features = self._extract_features(raw)
        if features is None:
            return

        self._window.append(features)
        self._records_since_check += 1

        if self._records_since_check >= DRIFT_WINDOW_SIZE and len(self._window) >= DRIFT_WINDOW_SIZE:
            self._records_since_check = 0
            await self._run_drift_check()

    def _extract_features(self, raw: dict) -> dict[str, float] | None:
        """
        Extract feature values from the raw payload.
        Returns None if essential fields are missing.
        """
        try:
            age_days = float(raw.get("age", 0))
            age_years = round(age_days / 365.25, 2)
            gender = int(raw.get("gender", 1))
            height_cm = int(raw.get("height", 170))
            weight_kg = float(raw.get("weight", 70))
            ap_hi = int(raw.get("ap_hi", 120))
            ap_lo = int(raw.get("ap_lo", 80))
            cholesterol = int(raw.get("cholesterol", 1))
            glucose = int(raw.get("gluc", 1))
            is_smoker = int(raw.get("smoke", 0))
            drinks_alcohol = int(raw.get("alco", 0))
            is_physically_active = int(raw.get("active", 1))

            bmi = round(weight_kg / ((height_cm / 100) ** 2), 4) if height_cm > 0 else 25.0
            pulse_pressure = ap_hi - ap_lo
            mean_arterial_pressure = ap_lo + pulse_pressure / 3.0

            # bp_category_encoded: 0=Normal, 1=Elevated, 2=HT Stage 1, 3=HT Stage 2, 4=Crisis
            if ap_hi < 120 and ap_lo < 80:
                bp_cat = 0
            elif ap_hi < 130 and ap_lo < 80:
                bp_cat = 1
            elif ap_hi < 140 or ap_lo < 90:
                bp_cat = 2
            elif ap_hi >= 180 or ap_lo >= 120:
                bp_cat = 4
            else:
                bp_cat = 3

            return {
                "age_years": age_years, "gender": gender,
                "height_cm": height_cm, "weight_kg": weight_kg,
                "ap_hi": ap_hi, "ap_lo": ap_lo,
                "cholesterol": cholesterol, "glucose": glucose,
                "is_smoker": is_smoker, "drinks_alcohol": drinks_alcohol,
                "is_physically_active": is_physically_active,
                "bmi": bmi, "pulse_pressure": pulse_pressure,
                "mean_arterial_pressure": mean_arterial_pressure,
                "bp_category_encoded": bp_cat,
            }
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            logger.debug("Feature extraction failed: %s", exc)
            return None

    async def _run_drift_check(self) -> None:
        """Run KS + PSI tests on the current window vs reference."""
        window_df = pd.DataFrame(list(self._window))

        ks_drifted: dict[str, float] = {}
        psi_scores: dict[str, float] = {}
        psi_drifted = False

        for feat in FEATURE_NAMES:
            if feat not in window_df.columns or feat not in self._reference.columns:
                continue

            ref_vals = self._reference[feat].dropna().values
            win_vals = window_df[feat].dropna().values

            if len(ref_vals) == 0 or len(win_vals) == 0:
                continue

            # KS test
            ks_stat, p_value = stats.ks_2samp(ref_vals, win_vals)
            if p_value < KS_P_THRESHOLD:
                ks_drifted[feat] = round(ks_stat, 6)

            # PSI
            psi = compute_psi(ref_vals.astype(float), win_vals.astype(float))
            psi_scores[feat] = round(psi, 6)
            if psi > PSI_THRESHOLD:
                psi_drifted = True

        ks_triggered = len(ks_drifted) >= KS_MIN_FEATURES
        drift_detected = ks_triggered or psi_drifted

        if drift_detected:
            logger.warning(
                "DRIFT DETECTED — KS drifted features: %d (%s), PSI breach: %s",
                len(ks_drifted), list(ks_drifted.keys()), psi_drifted,
            )
            event = ModelDriftDetected(
                drifted_features=ks_drifted,
                psi_scores=psi_scores,
                window_size=len(self._window),
                current_model_version=self._model_version,
            )
            await self._publisher.publish(event)
            logger.info("Published ModelDriftDetected event_id=%s", event.event_id)
        else:
            logger.info(
                "Drift check passed — KS drifted: %d/%d, max PSI: %.4f",
                len(ks_drifted), KS_MIN_FEATURES,
                max(psi_scores.values()) if psi_scores else 0.0,
            )


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    logger.info("DriftDetectionService starting...")

    # Load reference distribution once at startup
    reference_df = load_reference_distribution()

    publisher = RabbitMQPublisher()
    await publisher.connect()

    consumer = DriftConsumer(publisher=publisher, reference_df=reference_df)

    try:
        await consumer.start_consuming()
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    finally:
        await consumer.stop()
        await publisher.close()
        logger.info("DriftDetectionService stopped cleanly")


if __name__ == "__main__":
    asyncio.run(main())
