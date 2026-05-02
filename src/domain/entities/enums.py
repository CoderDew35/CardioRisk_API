"""
Domain enumerations for the CardioRisk system.

These enums map directly to the confirmed dataset encoding:
  gender:      1=Male, 2=Female
  cholesterol: 1=Normal, 2=Above Normal, 3=Well Above Normal
  gluc:        1=Normal, 2=Above Normal, 3=Well Above Normal
  bp_category: derived from (ap_hi, ap_lo) per ACC/AHA 2017 guidelines
"""
from enum import IntEnum, StrEnum


class Gender(IntEnum):
    MALE = 1
    FEMALE = 2


class CholesterolLevel(IntEnum):
    NORMAL = 1
    ABOVE_NORMAL = 2
    WELL_ABOVE_NORMAL = 3


class GlucoseLevel(IntEnum):
    NORMAL = 1
    ABOVE_NORMAL = 2
    WELL_ABOVE_NORMAL = 3


class BPCategory(StrEnum):
    """
    Blood pressure categories per ACC/AHA 2017 guidelines.
    Maps to the dataset's bp_category_encoded column.
    """
    NORMAL = "Normal"
    ELEVATED = "Elevated"
    HYPERTENSION_STAGE_1 = "Hypertension Stage 1"
    HYPERTENSION_STAGE_2 = "Hypertension Stage 2"
    HYPERTENSIVE_CRISIS = "Hypertensive Crisis"


class RiskLevel(StrEnum):
    """Human-readable risk band derived from the model's probability score."""
    LOW = "Low"          # 0.00 – 0.30
    MODERATE = "Moderate"  # 0.30 – 0.60
    HIGH = "High"          # 0.60 – 0.80
    VERY_HIGH = "Very High"  # 0.80 – 1.00
