"""
Unit tests — Domain Layer

Tests for:
  - PatientCardiovascularRecord construction + validation
  - BPClassifier categorisation
  - RiskScore value object
  - SHAPContribution
  - FeatureValidator
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from uuid import uuid4

from src.domain.entities.enums import BPCategory, CholesterolLevel, Gender, GlucoseLevel, RiskLevel
from src.domain.entities.patient_cardiovascular_record import PatientCardiovascularRecord
from src.domain.services.bp_classifier import BPClassifier
from src.domain.services.feature_validator import FeatureValidator
from src.domain.value_objects.risk_trajectory import RiskScore, SHAPContribution


#Fixtures ──────────

def make_record(**overrides) -> PatientCardiovascularRecord:
    defaults = dict(
        patient_id=uuid4(),
        recorded_at=datetime.now(timezone.utc),
        age_days=18393,
        gender=Gender.FEMALE,
        height_cm=168,
        weight_kg=62.0,
        ap_hi=110,
        ap_lo=80,
        cholesterol=CholesterolLevel.NORMAL,
        glucose=GlucoseLevel.NORMAL,
        is_smoker=False,
        drinks_alcohol=False,
        is_physically_active=True,
        age_years=50.35,
        bmi=21.97,
        bp_category=BPCategory.HYPERTENSION_STAGE_1,
    )
    defaults.update(overrides)
    return PatientCardiovascularRecord(**defaults)


#PatientCardiovascularRecord ─────────────────────────────────────────────

class TestPatientCardiovascularRecord:

    def test_valid_record_constructs(self):
        record = make_record()
        assert record.height_cm == 168
        assert record.bmi == pytest.approx(21.97, abs=0.01)

    def test_pulse_pressure_computed(self):
        record = make_record(ap_hi=120, ap_lo=80)
        assert record.pulse_pressure == 40

    def test_map_computed(self):
        record = make_record(ap_hi=120, ap_lo=80)
        # MAP = 80 + (40/3) ≈ 93.33
        assert record.mean_arterial_pressure == pytest.approx(93.33, abs=0.01)

    def test_is_hypertensive_stage1(self):
        record = make_record(bp_category=BPCategory.HYPERTENSION_STAGE_1)
        assert record.is_hypertensive is True

    def test_is_not_hypertensive_normal(self):
        record = make_record(bp_category=BPCategory.NORMAL)
        assert record.is_hypertensive is False

    def test_invalid_height_raises(self):
        with pytest.raises(ValueError, match="height_cm"):
            make_record(height_cm=999)

    def test_invalid_ap_lo_gt_ap_hi_raises(self):
        with pytest.raises(ValueError, match="Diastolic"):
            make_record(ap_hi=80, ap_lo=100)

    def test_feature_dict_has_required_keys(self):
        record = make_record()
        features = record.to_feature_dict()
        required = {"age_years", "bmi", "ap_hi", "ap_lo", "cholesterol", "bp_category_encoded"}
        assert required.issubset(features.keys())

    def test_immutability(self):
        record = make_record()
        with pytest.raises(Exception):  # frozen=True raises FrozenInstanceError
            record.age_days = 99999  # type: ignore


#BPClassifier ──────

class TestBPClassifier:

    @pytest.mark.parametrize("ap_hi,ap_lo,expected", [
        (110, 70, BPCategory.NORMAL),
        (125, 75, BPCategory.ELEVATED),
        (135, 85, BPCategory.HYPERTENSION_STAGE_1),
        (145, 95, BPCategory.HYPERTENSION_STAGE_2),
        (185, 125, BPCategory.HYPERTENSIVE_CRISIS),
    ])
    def test_classification(self, ap_hi, ap_lo, expected):
        assert BPClassifier.classify(ap_hi, ap_lo) == expected

    def test_encode_decode_roundtrip(self):
        for category in BPCategory:
            encoded = BPClassifier.encode(category)
            decoded = BPClassifier.from_encoded(encoded)
            assert decoded == category

    def test_invalid_ap_lo_raises(self):
        with pytest.raises(ValueError):
            BPClassifier.classify(80, 90)   # diastolic > systolic


#RiskScore ─────────

class TestRiskScore:

    def test_valid_score(self):
        s = RiskScore(0.73)
        assert s.percentage == pytest.approx(73.0)
        assert s.risk_level == RiskLevel.HIGH

    @pytest.mark.parametrize("value,expected_level", [
        (0.10, RiskLevel.LOW),
        (0.45, RiskLevel.MODERATE),
        (0.70, RiskLevel.HIGH),
        (0.90, RiskLevel.VERY_HIGH),
    ])
    def test_risk_levels(self, value, expected_level):
        assert RiskScore(value).risk_level == expected_level

    def test_invalid_score_raises(self):
        with pytest.raises(ValueError):
            RiskScore(1.5)

    def test_delta_positive(self):
        s1 = RiskScore(0.60)
        s2 = RiskScore(0.73)
        assert s2.delta(s1) == pytest.approx(0.13, abs=0.001)

    def test_delta_negative(self):
        s1 = RiskScore(0.73)
        s2 = RiskScore(0.60)
        assert s2.delta(s1) == pytest.approx(-0.13, abs=0.001)


#FeatureValidator ──

class TestFeatureValidator:

    def _valid_payload(self) -> dict:
        return {
            "age": 18393, "gender": 2, "height": 168, "weight": 62,
            "ap_hi": 110, "ap_lo": 80, "cholesterol": 1, "gluc": 1,
            "smoke": 0, "alco": 0, "active": 1,
        }

    def test_valid_payload_passes(self):
        result = FeatureValidator.validate(self._valid_payload())
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_invalid_ap_hi_fails(self):
        payload = self._valid_payload()
        payload["ap_hi"] = 999
        result = FeatureValidator.validate(payload)
        assert result.is_valid is False
        assert any("ap_hi" in e for e in result.errors)

    def test_diastolic_gt_systolic_fails(self):
        payload = self._valid_payload()
        payload["ap_hi"] = 80
        payload["ap_lo"] = 100
        result = FeatureValidator.validate(payload)
        assert result.is_valid is False

    def test_hypertensive_crisis_warning(self):
        payload = self._valid_payload()
        payload["ap_hi"] = 190
        result = FeatureValidator.validate(payload)
        assert result.is_valid is True
        assert len(result.warnings) > 0

    def test_invalid_height_fails(self):
        """Height bounds are now correctly checked against raw field name 'height'."""
        payload = self._valid_payload()
        payload["height"] = 999
        result = FeatureValidator.validate(payload)
        assert result.is_valid is False
        assert any("height" in e for e in result.errors)
