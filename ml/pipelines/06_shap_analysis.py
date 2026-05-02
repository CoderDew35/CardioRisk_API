"""
06_shap_analysis.py — make shap target orchestrator

Runs in sequence:
  1. Cohort SHAP (global feature importance)
  2. Temporal SHAP (per-patient trajectories with Δ-SHAP)
"""
from __future__ import annotations
import logging
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== Phase 6: SHAP Analysis ===")

    logger.info("--- Step 1/2: Global cohort SHAP ---")
    from ml.explainability.cohort_shap_analyzer import main as cohort_main
    cohort_main()

    logger.info("--- Step 2/2: Temporal SHAP trajectories ---")
    from ml.explainability.temporal_shap_aggregator import main as temporal_main
    temporal_main()

    logger.info("=== SHAP analysis complete ===")
    logger.info("Outputs:")
    logger.info("  ml/models/global_shap_summary.csv")
    logger.info("  ml/models/sample_trajectories.json")
    logger.info("  ml/models/temporal_shap_stats.csv")


if __name__ == "__main__":
    main()
