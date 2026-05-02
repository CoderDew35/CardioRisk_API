"""
Unit tests for Phase 6 — Temporal SHAP & Waterfall Builder

Tests run without any external services (no DB, no model file needed).
All model calls are mocked.

Success criteria:
  - Monte Carlo perturbation produces variance across steps
  - Δ-SHAP is correctly computed (None at T=0, float at T>0)
  - Waterfall builder output matches expected JSON schema
  - Cohort SHAP summary has correct structure
"""
from __future__ import annotations

import json
import math
import random
from unittest.mock import MagicMock

import pytest

#Waterfall builder (no deps) ─────────────────────────────────────────────
from ml.explainability.waterfall_builder import build_waterfall

#Perturbation logic (isolated, no model needed) ─────────────────────────
from ml.explainability.temporal_shap_aggregator import perturb_features, PERTURBATION_CONFIG


# ───────────────────────
# Fixtures
# ───────────────────────

BASE_FEATURES = {
    "age_years": 50.0, "gender": 2.0, "height_cm": 168.0, "weight_kg": 62.0,
    "ap_hi": 130.0, "ap_lo": 85.0, "cholesterol": 2.0, "glucose": 1.0,
    "is_smoker": 0.0, "drinks_alcohol": 0.0, "is_physically_active": 1.0,
    "bmi": 21.97, "pulse_pressure": 45.0, "mean_arterial_pressure": 100.0,
    "bp_category_encoded": 2.0,
}

SHAP_STEP_0 = [
    {"feature": "ap_hi",    "value": 130.0, "shap": 0.15,  "delta": None},
    {"feature": "age_years","value": 50.0,  "shap": 0.10,  "delta": None},
    {"feature": "bmi",      "value": 21.97, "shap": -0.05, "delta": None},
]

SHAP_STEP_1 = [
    {"feature": "ap_hi",    "value": 133.0, "shap": 0.19,  "delta": 0.04},
    {"feature": "age_years","value": 50.0,  "shap": 0.10,  "delta": 0.00},
    {"feature": "bmi",      "value": 22.10, "shap": -0.04, "delta": 0.01},
]


# ───────────────────────
# Perturbation tests
# ───────────────────────

class TestPerturbFeatures:

    def test_returns_new_dict(self):
        """perturb_features must not mutate the original dict."""
        original = dict(BASE_FEATURES)
        result = perturb_features(original, seed=42)
        assert result is not original

    def test_original_unchanged(self):
        original = dict(BASE_FEATURES)
        perturb_features(original, seed=42)
        assert original == BASE_FEATURES

    def test_time_varying_features_change(self):
        """After perturbation, at least one time-varying feature must differ."""
        result = perturb_features(BASE_FEATURES, seed=99)
        time_varying = list(PERTURBATION_CONFIG.keys())
        changed = any(
            result[f] != BASE_FEATURES[f]
            for f in time_varying if f in result
        )
        assert changed, "No features changed after perturbation"

    def test_static_features_unchanged(self):
        """Gender, cholesterol, glucose, smoke, alco, active never perturb."""
        result = perturb_features(BASE_FEATURES, seed=42)
        for static in ["gender", "cholesterol", "glucose", "is_smoker",
                       "drinks_alcohol", "is_physically_active"]:
            assert result[static] == BASE_FEATURES[static], \
                f"{static} should not be perturbed"

    def test_values_stay_within_bounds(self):
        """Perturbed values must respect clinical bounds."""
        random.seed(0)
        features = dict(BASE_FEATURES)
        for _ in range(20):
            features = perturb_features(features)
        for feat, cfg in PERTURBATION_CONFIG.items():
            if feat in features:
                assert cfg["min"] <= features[feat] <= cfg["max"], \
                    f"{feat}={features[feat]} exceeded bounds [{cfg['min']}, {cfg['max']}]"

    def test_reproducible_with_seed(self):
        r1 = perturb_features(BASE_FEATURES, seed=7)
        r2 = perturb_features(BASE_FEATURES, seed=7)
        assert r1 == r2

    def test_different_seeds_produce_different_results(self):
        r1 = perturb_features(BASE_FEATURES, seed=1)
        r2 = perturb_features(BASE_FEATURES, seed=2)
        # Very unlikely to be identical
        time_varying = list(PERTURBATION_CONFIG.keys())
        diffs = [r1[f] != r2[f] for f in time_varying if f in r1]
        assert any(diffs)

    def test_five_step_trajectory_shows_variance(self):
        """Over 5 perturbation steps, ap_hi should have variance > 0."""
        features = dict(BASE_FEATURES)
        ap_hi_values = [features["ap_hi"]]
        for i in range(5):
            features = perturb_features(features, seed=i)
            ap_hi_values.append(features["ap_hi"])
        # Variance across 6 values should be non-zero
        mean = sum(ap_hi_values) / len(ap_hi_values)
        variance = sum((x - mean) ** 2 for x in ap_hi_values) / len(ap_hi_values)
        assert variance > 0, f"ap_hi showed no variance across steps: {ap_hi_values}"

    def test_derived_features_consistent_after_perturbation(self):
        """BMI, pulse_pressure, MAP must be consistent with root features."""
        features = dict(BASE_FEATURES)
        for i in range(5):
            features = perturb_features(features, seed=i)
            # pulse_pressure = ap_hi - ap_lo
            expected_pp = features["ap_hi"] - features["ap_lo"]
            assert abs(features["pulse_pressure"] - expected_pp) < 0.01, \
                f"pulse_pressure inconsistent: {features['pulse_pressure']} != {expected_pp}"
            # MAP = ap_lo + (ap_hi - ap_lo) / 3
            expected_map = features["ap_lo"] + (features["ap_hi"] - features["ap_lo"]) / 3.0
            assert abs(features["mean_arterial_pressure"] - expected_map) < 0.01, \
                f"MAP inconsistent: {features['mean_arterial_pressure']} != {expected_map}"
            # ap_lo < ap_hi
            assert features["ap_lo"] < features["ap_hi"], \
                f"ap_lo ({features['ap_lo']}) >= ap_hi ({features['ap_hi']})"


# ───────────────────────
# Waterfall builder tests
# ───────────────────────

class TestBuildWaterfall:

    def test_returns_dict(self):
        result = build_waterfall(SHAP_STEP_0, risk_score=0.45, time_step=0)
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = build_waterfall(SHAP_STEP_0, risk_score=0.45, time_step=0)
        assert "time_step" in result
        assert "risk_score" in result
        assert "risk_pct" in result
        assert "bars" in result

    def test_bars_count_matches_contributions(self):
        result = build_waterfall(SHAP_STEP_0, risk_score=0.45, time_step=0)
        assert len(result["bars"]) == len(SHAP_STEP_0)

    def test_bar_structure(self):
        result = build_waterfall(SHAP_STEP_0, risk_score=0.45, time_step=0)
        bar = result["bars"][0]
        assert "feature" in bar
        assert "value" in bar
        assert "shap" in bar
        assert "direction" in bar  # "positive" | "negative" | "neutral"
        assert bar["direction"] in ("positive", "negative", "neutral")

    def test_direction_positive(self):
        result = build_waterfall(SHAP_STEP_0, risk_score=0.45, time_step=0)
        ap_hi_bar = next(b for b in result["bars"] if b["feature"] == "ap_hi")
        assert ap_hi_bar["direction"] == "positive"

    def test_direction_negative(self):
        result = build_waterfall(SHAP_STEP_0, risk_score=0.45, time_step=0)
        bmi_bar = next(b for b in result["bars"] if b["feature"] == "bmi")
        assert bmi_bar["direction"] == "negative"

    def test_delta_none_at_step_0(self):
        result = build_waterfall(SHAP_STEP_0, risk_score=0.45, time_step=0)
        for bar in result["bars"]:
            assert bar["delta"] is None

    def test_delta_present_at_step_1(self):
        result = build_waterfall(SHAP_STEP_1, risk_score=0.48, time_step=1)
        ap_hi_bar = next(b for b in result["bars"] if b["feature"] == "ap_hi")
        assert ap_hi_bar["delta"] == pytest.approx(0.04, abs=1e-6)

    def test_json_serialisable(self):
        """Output must be directly serialisable — no numpy types."""
        result = build_waterfall(SHAP_STEP_1, risk_score=0.48, time_step=1)
        serialised = json.dumps(result)   # raises TypeError if not serialisable
        assert len(serialised) > 0

    def test_risk_pct_is_percentage(self):
        result = build_waterfall(SHAP_STEP_0, risk_score=0.45, time_step=0)
        assert result["risk_pct"] == pytest.approx(45.0, abs=0.01)

    def test_empty_contributions(self):
        result = build_waterfall([], risk_score=0.5, time_step=0)
        assert result["bars"] == []
