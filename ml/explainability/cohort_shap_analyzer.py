"""
cohort_shap_analyzer.py

Computes global SHAP feature importance across the full dataset.
Outputs ml/models/global_shap_summary.csv with:
  - mean_abs_shap: average absolute impact per feature
  - std_abs_shap:  standard deviation
  - pct_positive:  % of patients where this feature increases risk

Used for the dashboard's global feature importance panel.

Run:  python ml/explainability/cohort_shap_analyzer.py
"""
from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="shap")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import joblib
import numpy as np
import pandas as pd
import sqlalchemy

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("ml/models")
FEATURE_NAMES = [
    "age_years", "gender", "height_cm", "weight_kg",
    "ap_hi", "ap_lo", "cholesterol", "glucose",
    "is_smoker", "drinks_alcohol", "is_physically_active",
    "bmi", "pulse_pressure", "mean_arterial_pressure", "bp_category_encoded",
]
# Analyse a representative sample — full 68K is slow for SHAP
SAMPLE_SIZE = int(os.getenv("COHORT_SHAP_SAMPLE_SIZE", "2000"))


def load_sample(n: int) -> pd.DataFrame:
    url = os.getenv("DATABASE_URL", "").replace("postgresql+asyncpg://", "postgresql://")
    engine = sqlalchemy.create_engine(url)
    query = f"""
        SELECT
            age_years, gender, height_cm, weight_kg,
            ap_hi, ap_lo, cholesterol, glucose,
            is_smoker::int, drinks_alcohol::int, is_physically_active::int,
            bmi, bp_category_encoded,
            has_cardiovascular_disease::int AS label
        FROM patient_cardiovascular_records
        WHERE has_cardiovascular_disease IS NOT NULL
          AND data_source = 'batch_csv'
        ORDER BY RANDOM()
        LIMIT {n}
    """
    df = pd.read_sql(query, engine)
    engine.dispose()
    df["pulse_pressure"]         = df["ap_hi"] - df["ap_lo"]
    df["mean_arterial_pressure"] = df["ap_lo"] + df["pulse_pressure"] / 3.0
    logger.info("Loaded %d patients for cohort SHAP analysis", len(df))
    return df


def _extract_shap_matrix(shap_values) -> np.ndarray:
    """Normalise shap_values output to shape (n_samples, n_features)."""
    if isinstance(shap_values, list):
        return shap_values[1]            # old format: list[class][samples, features]
    if hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        return shap_values[:, :, 1]      # new format: (samples, features, classes)
    return shap_values


def analyze(model, explainer, df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_NAMES]
    logger.info("Computing SHAP values for %d patients...", len(X))
    raw = explainer.shap_values(X)
    shap_matrix = _extract_shap_matrix(raw)   # shape: (n, 15)

    abs_shap = np.abs(shap_matrix)
    summary = pd.DataFrame({
        "feature":       FEATURE_NAMES,
        "mean_abs_shap": abs_shap.mean(axis=0),
        "std_abs_shap":  abs_shap.std(axis=0),
        "pct_positive":  (shap_matrix > 0).mean(axis=0) * 100,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return summary


def main() -> None:
    model_path     = MODELS_DIR / "lgbm_cardio_v1.joblib"
    explainer_path = MODELS_DIR / "shap_explainer_v1.joblib"

    if not model_path.exists() or not explainer_path.exists():
        raise FileNotFoundError("Run 'make train' first.")

    model    = joblib.load(model_path)
    explainer = joblib.load(explainer_path)

    df      = load_sample(SAMPLE_SIZE)
    summary = analyze(model, explainer, df)

    out = MODELS_DIR / "global_shap_summary.csv"
    summary.to_csv(out, index=False)
    logger.info("Global SHAP summary saved → %s", out)
    logger.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
