"""
08_simulate_stream.py — Thesis Defence Demo Script

Simulates the full MLOps Continuous Training cycle end-to-end:

  Phase 1 (clean stream):   Records 20K→48K streamed via /ingest. No drift.
  Phase 2 (drift injection): Records 48K→68K with ap_hi += 20, weight_kg += 15.
  Phase 3 (drift trigger):   DriftDetectionService detects KS divergence.
  Phase 4 (auto retrain):    CTService retrains, evaluates, promotes, API hot-swaps.

Prerequisites:
  - docker compose up (RabbitMQ, Postgres, MinIO, MLflow)
  - make seed-db (data loaded)
  - make train  (champion model registered)
  - make dev    (API running)
  - make drift  (DriftDetectionService running)
  - make ct     (ContinuousTrainingService running)

Run: make simulate-stream
     python ml/pipelines/08_simulate_stream.py
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | SimulateStream | %(message)s",
)
logger = logging.getLogger(__name__)

# Config
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
INGEST_URL = f"{API_BASE}/v1/patients/ingest"
MLOPS_URL = f"{API_BASE}/v1/mlops/status"
BATCH_SIZE = int(os.getenv("STREAM_BATCH_SIZE", "100"))
SLEEP_BETWEEN_BATCHES = float(os.getenv("STREAM_SLEEP", "0.1"))

# Dataset boundaries
CLEAN_START = 20_000
CLEAN_END = 48_000
DRIFT_START = 48_000
DRIFT_END = 68_000

# Drift injection parameters
DRIFT_AP_HI_DELTA = 20    # mmHg increase
DRIFT_WEIGHT_DELTA = 15   # kg increase


def load_raw_dataset() -> pd.DataFrame:
    """Load the original CSV dataset."""
    csv_path = os.getenv("DATASET_CSV_PATH", "ml/data/cardio_dataset.csv")
    if not os.path.exists(csv_path):
        logger.error("Dataset not found at %s", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    logger.info("Loaded dataset: %d records", len(df))
    return df


def row_to_ingest_payload(row: pd.Series) -> dict:
    """Convert a raw CSV row to the ingest API payload format."""
    return {
        "raw_payload": {
            "age": int(row.get("age", 0)),
            "gender": int(row.get("gender", 1)),
            "height": int(row.get("height", 170)),
            "weight": float(row.get("weight", 70)),
            "ap_hi": int(row.get("ap_hi", 120)),
            "ap_lo": int(row.get("ap_lo", 80)),
            "cholesterol": int(row.get("cholesterol", 1)),
            "gluc": int(row.get("gluc", 1)),
            "smoke": int(row.get("smoke", 0)),
            "alco": int(row.get("alco", 0)),
            "active": int(row.get("active", 1)),
            "cardio": int(row.get("cardio", 0)),
        },
        "source": "stream_simulation",
    }


def inject_drift(row: pd.Series) -> pd.Series:
    """Apply artificial drift to a row: inflate ap_hi and weight."""
    row = row.copy()
    row["ap_hi"] = int(row["ap_hi"]) + DRIFT_AP_HI_DELTA
    row["weight"] = float(row["weight"]) + DRIFT_WEIGHT_DELTA
    return row


def stream_batch(df_slice: pd.DataFrame, label: str, apply_drift: bool = False) -> int:
    """Stream a slice of the dataset through the ingest API in batches."""
    total = len(df_slice)
    success_count = 0
    error_count = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = df_slice.iloc[batch_start:batch_end]

        for _, row in batch.iterrows():
            if apply_drift:
                row = inject_drift(row)

            payload = row_to_ingest_payload(row)
            try:
                resp = requests.post(INGEST_URL, json=payload, timeout=5)
                if resp.status_code in (200, 202):
                    success_count += 1
                else:
                    error_count += 1
            except requests.RequestException as exc:
                error_count += 1
                if error_count <= 3:
                    logger.warning("Ingest request failed: %s", exc)

        logger.info(
            "[%s] Batch %d-%d/%d sent (ok=%d, err=%d)",
            label, batch_start, batch_end, total, success_count, error_count,
        )
        time.sleep(SLEEP_BETWEEN_BATCHES)

    return success_count


def check_mlops_status() -> dict | None:
    """Query the MLOps status endpoint."""
    try:
        resp = requests.get(MLOPS_URL, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return None


def main() -> None:
    logger.info("=" * 60)
    logger.info("CardioRisk Thesis Demo — Simulated Streaming + Drift Injection")
    logger.info("=" * 60)

    # Check API is running
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=3)
        if resp.status_code != 200:
            logger.error("API not responding at %s", API_BASE)
            sys.exit(1)
    except requests.RequestException:
        logger.error("Cannot reach API at %s — is it running?", API_BASE)
        sys.exit(1)

    logger.info("API is healthy ✓")

    # Check current model
    status = check_mlops_status()
    if status:
        logger.info("Current model: %s", status.get("current_model_version", "unknown"))

    # Load dataset
    df = load_raw_dataset()

    if len(df) < DRIFT_END:
        logger.warning(
            "Dataset has only %d records — adjusting boundaries", len(df)
        )

    # ── Phase 1: Clean Stream ────────────────────────────────────────────
    logger.info("")
    logger.info("═══ Phase 1: Clean Stream (records %d→%d) ═══", CLEAN_START, CLEAN_END)
    clean_slice = df.iloc[CLEAN_START:min(CLEAN_END, len(df))]
    t0 = time.time()
    clean_count = stream_batch(clean_slice, "CLEAN", apply_drift=False)
    logger.info(
        "Phase 1 complete: %d records in %.1fs",
        clean_count, time.time() - t0,
    )

    # Brief pause to let DriftService process the window
    logger.info("Waiting 3s for drift service to process...")
    time.sleep(3)

    # ── Phase 2: Drift Injection ─────────────────────────────────────────
    logger.info("")
    logger.info(
        "═══ Phase 2: Drift Injection (records %d→%d, ap_hi+%d, weight+%d) ═══",
        DRIFT_START, DRIFT_END, DRIFT_AP_HI_DELTA, DRIFT_WEIGHT_DELTA,
    )
    drift_slice = df.iloc[DRIFT_START:min(DRIFT_END, len(df))]
    t0 = time.time()
    drift_count = stream_batch(drift_slice, "DRIFTED", apply_drift=True)
    logger.info(
        "Phase 2 complete: %d records in %.1fs",
        drift_count, time.time() - t0,
    )

    # ── Phase 3 & 4: Wait for Drift Detection + Retraining ──────────────
    logger.info("")
    logger.info("═══ Phase 3-4: Waiting for Drift Detection → CT Cycle ═══")
    logger.info(
        "DriftDetectionService should detect KS divergence in ap_hi and weight_kg."
    )
    logger.info(
        "ContinuousTrainingService should retrain and hot-swap the model."
    )
    logger.info("")
    logger.info("Monitor the following:")
    logger.info("  - DriftService terminal: watch for 'DRIFT DETECTED'")
    logger.info("  - CTService terminal:    watch for 'PROMOTED' or 'NOT PROMOTED'")
    logger.info("  - MLflow UI:             http://localhost:5050")
    logger.info("  - API MLOps status:      curl %s", MLOPS_URL)

    # Poll for model version change
    original_version = status.get("current_model_version", "unknown") if status else "unknown"
    logger.info("")
    logger.info("Polling for model version change (timeout 5 minutes)...")

    poll_start = time.time()
    POLL_TIMEOUT = 300  # 5 minutes

    while time.time() - poll_start < POLL_TIMEOUT:
        new_status = check_mlops_status()
        if new_status:
            new_version = new_status.get("current_model_version", "unknown")
            if new_version != original_version and new_version != "unknown":
                logger.info("")
                logger.info("═══ MODEL HOT-SWAPPED ═══")
                logger.info("  Old version: %s", original_version)
                logger.info("  New version: %s", new_version)
                logger.info("═══ Thesis demo complete! ═══")
                return
        time.sleep(5)

    logger.warning(
        "Polling timed out after %ds — check service logs for status.",
        POLL_TIMEOUT,
    )
    logger.info("The CT cycle may still be running. Check:")
    logger.info("  curl %s", MLOPS_URL)
    logger.info("  open http://localhost:5050")


if __name__ == "__main__":
    main()
