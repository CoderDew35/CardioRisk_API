"""
LightGBMAdapter — implements IRiskModel port.

Loads a pre-trained LightGBM model from disk (joblib serialized).
Returns RiskScore value objects. Stateless after loading.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.domain.value_objects.risk_trajectory import RiskScore

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "ml/models/lgbm_cardio_v1.joblib")

# Feature order must match training pipeline exactly
FEATURE_NAMES = [
    "age_years", "gender", "height_cm", "weight_kg",
    "ap_hi", "ap_lo", "cholesterol", "glucose",
    "is_smoker", "drinks_alcohol", "is_physically_active",
    "bmi", "pulse_pressure", "mean_arterial_pressure", "bp_category_encoded",
]


class ModelNotLoadedError(Exception):
    pass


class LightGBMAdapter:
    """
    Wraps a scikit-learn/LightGBM pipeline.
    Implements IRiskModel — call predict(features_dict) → RiskScore.
    """

    def __init__(self, model_path: str = MODEL_PATH) -> None:
        self._model_path = model_path
        self._model = None
        self._version = "not_loaded"
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._model_path):
            logger.warning(
                "Model file not found at %s. "
                "Run 'make train' to generate the model. "
                "Adapter will return 0.5 (random) until model is loaded.",
                self._model_path,
            )
            return

        self._model = joblib.load(self._model_path)
        # Extract version from filename (e.g., lgbm_cardio_v1 → v1)
        stem = os.path.splitext(os.path.basename(self._model_path))[0]
        self._version = stem.split("_")[-1]
        logger.info("Model loaded: %s (version=%s)", self._model_path, self._version)

    def predict(self, features: dict[str, float]) -> RiskScore:
        return RiskScore(self.predict_proba(features))

    def predict_proba(self, features: dict[str, float]) -> float:
        if self._model is None:
            logger.warning("Model not loaded — returning default score 0.5")
            return 0.5

        df = pd.DataFrame([features])[FEATURE_NAMES]
        proba: float = float(self._model.predict_proba(df)[0][1])
        return round(proba, 6)

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def feature_names(self) -> list[str]:
        return FEATURE_NAMES
