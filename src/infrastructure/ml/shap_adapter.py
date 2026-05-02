"""
SHAPTreeExplainerAdapter — implements IExplainer port.

Uses shap.TreeExplainer for exact, fast SHAP values on tree-based models.
Computes Δ-SHAP when previous contributions are provided (temporal comparison).
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd

from src.domain.value_objects.risk_trajectory import RiskScore, SHAPContribution

if TYPE_CHECKING:
    from src.infrastructure.ml.lightgbm_adapter import LightGBMAdapter

logger = logging.getLogger(__name__)

SHAP_EXPLAINER_PATH = os.getenv("SHAP_EXPLAINER_PATH", "ml/models/shap_explainer_v1.joblib")


class SHAPTreeExplainerAdapter:
    """
    Implements IExplainer using shap.TreeExplainer.

    Initialisation options:
      1. Load pre-saved explainer from disk (fastest — use in production)
      2. Build explainer from LightGBMAdapter model object (use during training)
    """

    def __init__(
        self,
        model: "LightGBMAdapter | None" = None,
        explainer_path: str = SHAP_EXPLAINER_PATH,
    ) -> None:
        self._explainer = None
        self._feature_names: list[str] = []

        if os.path.exists(explainer_path):
            self._explainer = joblib.load(explainer_path)
            logger.info("SHAP explainer loaded from %s", explainer_path)
        elif model is not None and model._model is not None:
            import shap
            self._explainer = shap.TreeExplainer(model._model)
            self._feature_names = model.feature_names
            logger.info("SHAP explainer built from model object")
        else:
            logger.warning(
                "SHAP explainer not available. "
                "Run 'make shap' to generate. Returning zero SHAP values."
            )

    def explain(
        self,
        features: dict[str, float],
        risk_score: RiskScore,
        previous_contributions: list[SHAPContribution] | None = None,
    ) -> list[SHAPContribution]:
        """
        Args:
            features:              Feature dict for this time step
            risk_score:            Predicted risk (used for context only)
            previous_contributions: T-1 contributions for Δ-SHAP calculation

        Returns:
            List of SHAPContribution, sorted by |shap_value| descending
        """
        feature_names = self._feature_names or list(features.keys())
        df = pd.DataFrame([features])

        if self._explainer is None:
            # Return zeroed contributions if explainer unavailable
            return [
                SHAPContribution(
                    feature_name=f,
                    feature_value=float(features.get(f, 0)),
                    shap_value=0.0,
                    delta_from_previous=None,
                )
                for f in feature_names
            ]

        shap_values = self._explainer.shap_values(df)

        # Handle both old list format and new ndarray format (shap >= 0.44)
        if isinstance(shap_values, list):
            shap_row = shap_values[1][0]   # positive class (cardio=1), old format
        elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            shap_row = shap_values[0, :, 1]  # new format: (samples, features, classes)
        else:
            shap_row = shap_values[0]

        # Build previous contribution lookup for Δ-SHAP
        prev_lookup: dict[str, float] = {}
        if previous_contributions:
            prev_lookup = {c.feature_name: c.shap_value for c in previous_contributions}

        contributions = []
        cols = list(df.columns)
        for i, feat_name in enumerate(cols):
            shap_val = float(shap_row[i]) if i < len(shap_row) else 0.0
            prev_val = prev_lookup.get(feat_name)
            delta = round(shap_val - prev_val, 6) if prev_val is not None else None

            contributions.append(
                SHAPContribution(
                    feature_name=feat_name,
                    feature_value=float(features.get(feat_name, 0)),
                    shap_value=round(shap_val, 6),
                    delta_from_previous=delta,
                )
            )

        # Sort by absolute SHAP value (most impactful first)
        contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)
        return contributions
