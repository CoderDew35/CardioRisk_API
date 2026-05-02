"""
temporal_shap_aggregator.py

Batch pipeline script: loads N random patients from PostgreSQL,
runs Monte Carlo temporal SHAP trajectories (T=5 steps), and saves:
  - ml/models/sample_trajectories.json   (per-patient trajectory data)
  - ml/models/temporal_shap_stats.csv   (population Δ-SHAP statistics)

Run via:  make shap
          python ml/explainability/temporal_shap_aggregator.py
"""
from __future__ import annotations

import json
import logging
import os
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="shap")

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import joblib
import numpy as np
import pandas as pd
import shap as shap_lib
import sqlalchemy

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

#Config ─────────────
MODELS_DIR   = Path("ml/models")
N_PATIENTS   = int(os.getenv("TEMPORAL_SHAP_N_PATIENTS", "50"))
N_STEPS      = int(os.getenv("TEMPORAL_SHAP_N_STEPS", "5"))
RANDOM_STATE = 42

FEATURE_NAMES = [
    "age_years", "gender", "height_cm", "weight_kg",
    "ap_hi", "ap_lo", "cholesterol", "glucose",
    "is_smoker", "drinks_alcohol", "is_physically_active",
    "bmi", "pulse_pressure", "mean_arterial_pressure", "bp_category_encoded",
]

# Time-varying features and their perturbation config.
# Only ROOT features perturbed; derived features re-computed for consistency.
PERTURBATION_CONFIG: dict[str, dict] = {
    "ap_hi":                  {"std": 3.0,  "min": 60,   "max": 250},
    "ap_lo":                  {"std": 2.0,  "min": 40,   "max": 200},
    "weight_kg":              {"std": 0.5,  "min": 30,   "max": 200},
}


def perturb_features(features: dict, seed: int | None = None) -> dict:
    """
    Apply Gaussian noise to time-varying features.
    Returns a new dict; never mutates the input.

    Args:
        features: current feature dict
        seed:     optional RNG seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
    result = dict(features)
    for feat, cfg in PERTURBATION_CONFIG.items():
        if feat in result:
            noise = random.gauss(0, cfg["std"])
            result[feat] = max(cfg["min"], min(cfg["max"], result[feat] + noise))

    # Re-derive dependent features from perturbed roots
    ap_hi = result.get("ap_hi", 120)
    ap_lo = result.get("ap_lo", 80)
    if ap_lo >= ap_hi:
        ap_lo = ap_hi - 10
        result["ap_lo"] = ap_lo
    result["pulse_pressure"] = ap_hi - ap_lo
    result["mean_arterial_pressure"] = ap_lo + (ap_hi - ap_lo) / 3.0

    height_cm = result.get("height_cm", 170)
    weight_kg = result.get("weight_kg", 70)
    if height_cm > 0:
        result["bmi"] = round(weight_kg / ((height_cm / 100) ** 2), 4)

    return result


#Database helpers ───

def _make_engine():
    url = os.getenv("DATABASE_URL", "").replace("postgresql+asyncpg://", "postgresql://")
    return sqlalchemy.create_engine(url)


def load_sample_patients(n: int) -> pd.DataFrame:
    engine = _make_engine()
    query = f"""
        SELECT
            patient_id::text,
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
    # Derive engineered features
    df["pulse_pressure"]         = df["ap_hi"] - df["ap_lo"]
    df["mean_arterial_pressure"] = df["ap_lo"] + df["pulse_pressure"] / 3.0
    logger.info("Loaded %d sample patients from PostgreSQL", len(df))
    return df


#SHAP helpers ───────

def _extract_shap_row(shap_values, idx: int = 0) -> np.ndarray:
    """Handle both old list format and new ndarray format from shap library."""
    if isinstance(shap_values, list):
        return shap_values[1][idx]
    if hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        return shap_values[idx, :, 1]
    return shap_values[idx]


def compute_shap(explainer, features_df: pd.DataFrame) -> list[dict]:
    """
    Returns a list of {feature, shap} dicts for one row.
    """
    vals = explainer.shap_values(features_df)
    row = _extract_shap_row(vals, idx=0)
    return [
        {"feature": feat, "shap": float(row[i])}
        for i, feat in enumerate(features_df.columns)
    ]


#Trajectory runner ───

def run_trajectory(
    model,
    explainer,
    base_features: dict,
    patient_id: str,
    n_steps: int = N_STEPS,
    seed: int = RANDOM_STATE,
) -> list[dict]:
    """
    Runs a Monte Carlo trajectory for one patient.

    Returns list of step dicts:
      [{"time_step": 0, "risk_score": 0.45, "shap": {...}, "delta_shap": {...}}, ...]
    """
    steps = []
    current = dict(base_features)
    prev_shap: dict[str, float] | None = None

    for t in range(n_steps + 1):
        df = pd.DataFrame([current])[FEATURE_NAMES]
        risk_proba = float(model.predict_proba(df)[0][1])
        shap_contribs = compute_shap(explainer, df)
        shap_dict = {c["feature"]: c["shap"] for c in shap_contribs}

        delta_shap = None
        if prev_shap is not None:
            delta_shap = {
                feat: round(shap_dict[feat] - prev_shap[feat], 6)
                for feat in shap_dict
            }

        steps.append({
            "time_step":  t,
            "risk_score": round(risk_proba, 6),
            "risk_pct":   round(risk_proba * 100, 2),
            "features":   {k: round(v, 4) for k, v in current.items()
                           if k in FEATURE_NAMES},
            "shap":       {k: round(v, 6) for k, v in shap_dict.items()},
            "delta_shap": delta_shap,
        })

        prev_shap = shap_dict
        if t < n_steps:
            current = perturb_features(current, seed=seed + t)

    return steps


#Population-level Δ-SHAP stats ────────────────────────────────────────────

def compute_population_delta_shap(all_trajectories: list[dict]) -> pd.DataFrame:
    """
    For each feature, compute mean absolute Δ-SHAP across all patients and steps.
    High values = feature that drifts most over time (most dynamic risk driver).
    """
    records = []
    for traj in all_trajectories:
        for step in traj["steps"][1:]:   # skip T=0 (no delta)
            if step["delta_shap"] is None:
                continue
            for feat, delta in step["delta_shap"].items():
                records.append({"feature": feat, "abs_delta_shap": abs(delta)})

    if not records:
        return pd.DataFrame(columns=["feature", "mean_abs_delta_shap", "std_abs_delta_shap"])

    df = pd.DataFrame(records)
    summary = (
        df.groupby("feature")["abs_delta_shap"]
        .agg(mean_abs_delta_shap="mean", std_abs_delta_shap="std")
        .reset_index()
        .sort_values("mean_abs_delta_shap", ascending=False)
    )
    return summary


#Main ───────────────

def main() -> None:
    model_path    = MODELS_DIR / "lgbm_cardio_v1.joblib"
    explainer_path = MODELS_DIR / "shap_explainer_v1.joblib"

    if not model_path.exists() or not explainer_path.exists():
        raise FileNotFoundError(
            "Model or SHAP explainer not found. Run 'make train' first."
        )

    logger.info("Loading model and SHAP explainer...")
    model    = joblib.load(model_path)
    explainer = joblib.load(explainer_path)

    patients_df = load_sample_patients(N_PATIENTS)

    all_trajectories = []
    for i, row in patients_df.iterrows():
        patient_id = str(row["patient_id"])
        base = {feat: float(row[feat]) for feat in FEATURE_NAMES if feat in row.index}

        steps = run_trajectory(
            model, explainer, base,
            patient_id=patient_id,
            n_steps=N_STEPS,
            seed=RANDOM_STATE + i,
        )

        all_trajectories.append({
            "patient_id": patient_id,
            "true_label": int(row["label"]),
            "n_steps":    N_STEPS,
            "steps":      steps,
        })

        if (i + 1) % 10 == 0:
            logger.info("Processed %d / %d patients", i + 1, len(patients_df))

    # Save trajectory data
    out_json = MODELS_DIR / "sample_trajectories.json"
    with open(out_json, "w") as f:
        json.dump(all_trajectories, f, indent=2)
    logger.info("Saved %d trajectories → %s", len(all_trajectories), out_json)

    # Population Δ-SHAP stats
    summary_df = compute_population_delta_shap(all_trajectories)
    out_csv = MODELS_DIR / "temporal_shap_stats.csv"
    summary_df.to_csv(out_csv, index=False)
    logger.info("Saved population Δ-SHAP stats → %s", out_csv)
    logger.info("\nTop time-varying risk features:\n%s",
                summary_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
