"""
04_train_lgbm.py — LightGBM Training Pipeline with Optuna HPO

Pipeline:
  1. Load data from PostgreSQL (already seeded)
  2. Feature engineering (pulse_pressure, MAP)
  3. Train/val/test split (70/15/15, stratified)
  4. Baseline models (LogReg, RF) for comparison
  5. LightGBM with Optuna Bayesian HPO (50 trials)
  6. Final model evaluation + calibration
  7. Save model + SHAP explainer to ml/models/

Run: make train
     or: python ml/pipelines/04_train_lgbm.py
"""
from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

# Load .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

#Config ────────────
MODELS_DIR = Path("ml/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    "age_years", "gender", "height_cm", "weight_kg",
    "ap_hi", "ap_lo", "cholesterol", "glucose",
    "is_smoker", "drinks_alcohol", "is_physically_active",
    "bmi", "pulse_pressure", "mean_arterial_pressure", "bp_category_encoded",
]
TARGET = "has_cardiovascular_disease"
N_TRIALS = 50
RANDOM_STATE = 42


#1. Load data from PostgreSQL ─────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load seeded patient records directly from PostgreSQL."""
    import asyncio
    import asyncpg

    db_url = os.getenv("DATABASE_URL", "")
    # Convert asyncpg URL for sync psycopg2 use in pandas
    sync_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

    import sqlalchemy
    engine = sqlalchemy.create_engine(sync_url)
    query = """
        SELECT
            age_years, gender, height_cm, weight_kg,
            ap_hi, ap_lo, cholesterol, glucose,
            is_smoker::int, drinks_alcohol::int, is_physically_active::int,
            bmi, bp_category_encoded,
            has_cardiovascular_disease::int AS has_cardiovascular_disease
        FROM patient_cardiovascular_records
        WHERE has_cardiovascular_disease IS NOT NULL
          AND data_source = 'batch_csv'
    """
    df = pd.read_sql(query, engine)
    engine.dispose()
    logger.info("Loaded %d records from PostgreSQL", len(df))
    return df


#2. Feature engineering ───────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["mean_arterial_pressure"] = df["ap_lo"] + (df["pulse_pressure"] / 3.0)
    # Cast bool columns to int
    for col in ["is_smoker", "drinks_alcohol", "is_physically_active"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


#3. Train/val/test split ──────────────────────────────────────────────────

def split_data(df: pd.DataFrame):
    X = df[FEATURE_NAMES]
    y = df[TARGET]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, stratify=y_temp, random_state=RANDOM_STATE
    )
    logger.info(
        "Split: train=%d  val=%d  test=%d  (target rate=%.1f%%)",
        len(X_train), len(X_val), len(X_test),
        y.mean() * 100,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


#4. Baseline models ─

def train_baselines(X_train, y_train, X_val, y_val) -> None:
    logger.info("--- Baseline Models ---")

    # Logistic Regression
    lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    lr.fit(X_train, y_train)
    lr_auc = roc_auc_score(y_val, lr.predict_proba(X_val)[:, 1])
    logger.info("LogReg   AUC-ROC=%.4f", lr_auc)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
    logger.info("RandomForest AUC-ROC=%.4f", rf_auc)


#5. Optuna HPO for LightGBM ───────────────────────────────────────────────

def optuna_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "objective":        "binary",
        "metric":           "auc",
        "verbosity":        -1,
        "boosting_type":    "gbdt",
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 20, 150),
        "max_depth":        trial.suggest_int("max_depth", 3, 12),
        "min_child_samples":trial.suggest_int("min_child_samples", 10, 100),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state":     RANDOM_STATE,
        "n_jobs":           -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)


def train_lgbm_with_hpo(X_train, y_train, X_val, y_val) -> lgb.LGBMClassifier:
    logger.info("--- LightGBM Optuna HPO (%d trials) ---", N_TRIALS)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_params.update({
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    })

    logger.info("Best AUC-ROC (val): %.4f", study.best_value)
    logger.info("Best params: %s", best_params)

    # Retrain on train + val combined with best params
    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_full, y_full)

    return final_model


#6. Evaluate ────────

def evaluate(model, X_test, y_test, label: str = "LightGBM") -> dict:
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds = (preds_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, preds_proba)
    auprc = average_precision_score(y_test, preds_proba)
    brier = brier_score_loss(y_test, preds_proba)

    logger.info("=== %s Test Results ===", label)
    logger.info("  AUC-ROC : %.4f", auc)
    logger.info("  AUPRC   : %.4f", auprc)
    logger.info("  Brier   : %.4f", brier)
    print(classification_report(y_test, preds, target_names=["No CVD", "CVD"]))

    return {"auc_roc": auc, "auprc": auprc, "brier_score": brier}


#7. Save model + SHAP explainer ──────────────────────────────────────────

def save_artifacts(model: lgb.LGBMClassifier, X_train: pd.DataFrame) -> None:
    model_path = MODELS_DIR / "lgbm_cardio_v1.joblib"
    joblib.dump(model, model_path)
    logger.info("Model saved: %s", model_path)

    # Build and save SHAP TreeExplainer
    logger.info("Building SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    explainer_path = MODELS_DIR / "shap_explainer_v1.joblib"
    joblib.dump(explainer, explainer_path)
    logger.info("SHAP explainer saved: %s", explainer_path)

    # Save feature importance
    importance_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)

    importance_path = MODELS_DIR / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info("Feature importance saved: %s", importance_path)
    logger.info("\nTop 10 features by gain:\n%s", importance_df.head(10).to_string(index=False))


#Main ───────────────

def main() -> None:
    logger.info("=== CardioRisk LightGBM Training Pipeline ===")

    df = load_data()
    df = engineer_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    train_baselines(X_train, y_train, X_val, y_val)

    model = train_lgbm_with_hpo(X_train, y_train, X_val, y_val)

    metrics = evaluate(model, X_test, y_test)

    # Assert minimum quality bar
    assert metrics["auc_roc"] >= 0.70, f"AUC-ROC {metrics['auc_roc']:.4f} below threshold 0.70"
    logger.info("Quality bar passed: AUC-ROC=%.4f >= 0.70", metrics["auc_roc"])

    save_artifacts(model, X_train)
    logger.info("=== Training complete ===")


if __name__ == "__main__":
    main()
