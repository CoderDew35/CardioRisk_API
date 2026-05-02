"""
BPClassifier — pure domain service for blood pressure categorisation.

Implements ACC/AHA 2017 hypertension guidelines.
Used during ingestion to derive bp_category from raw ap_hi / ap_lo values.
Zero external dependencies.
"""
from src.domain.entities.enums import BPCategory


class BPClassifier:
    """
    Stateless domain service. All methods are static — no instantiation required.

    ACC/AHA 2017 Guidelines:
        Normal:               systolic < 120  AND  diastolic < 80
        Elevated:             systolic 120–129  AND  diastolic < 80
        Hypertension Stage 1: systolic 130–139  OR   diastolic 80–89
        Hypertension Stage 2: systolic ≥ 140   OR   diastolic ≥ 90
        Hypertensive Crisis:  systolic > 180   OR   diastolic > 120
    """

    @staticmethod
    def classify(ap_hi: int, ap_lo: int) -> BPCategory:
        """
        Classify blood pressure into ACC/AHA 2017 category.

        Args:
            ap_hi: Systolic blood pressure (mmHg)
            ap_lo: Diastolic blood pressure (mmHg)

        Returns:
            BPCategory enum value

        Raises:
            ValueError: If values are clinically implausible
        """
        if ap_hi <= 0 or ap_lo <= 0:
            raise ValueError("Blood pressure values must be positive")
        if ap_lo >= ap_hi:
            raise ValueError("Diastolic must be less than systolic")

        # Hypertensive crisis takes precedence
        if ap_hi > 180 or ap_lo > 120:
            return BPCategory.HYPERTENSIVE_CRISIS

        if ap_hi >= 140 or ap_lo >= 90:
            return BPCategory.HYPERTENSION_STAGE_2

        if ap_hi >= 130 or ap_lo >= 80:
            return BPCategory.HYPERTENSION_STAGE_1

        if 120 <= ap_hi <= 129 and ap_lo < 80:
            return BPCategory.ELEVATED

        return BPCategory.NORMAL

    @staticmethod
    def encode(category: BPCategory) -> int:
        """
        Integer encoding matching the dataset's bp_category_encoded column.
        Normal=0, Elevated=1, Stage1=2, Stage2=3, Crisis=4
        """
        _mapping = {
            BPCategory.NORMAL: 0,
            BPCategory.ELEVATED: 1,
            BPCategory.HYPERTENSION_STAGE_1: 2,
            BPCategory.HYPERTENSION_STAGE_2: 3,
            BPCategory.HYPERTENSIVE_CRISIS: 4,
        }
        return _mapping[category]

    @staticmethod
    def from_encoded(encoded: int) -> BPCategory:
        """Reverse mapping from integer encoding back to BPCategory."""
        _mapping = {
            0: BPCategory.NORMAL,
            1: BPCategory.ELEVATED,
            2: BPCategory.HYPERTENSION_STAGE_1,
            3: BPCategory.HYPERTENSION_STAGE_2,
            4: BPCategory.HYPERTENSIVE_CRISIS,
        }
        if encoded not in _mapping:
            raise ValueError(f"Unknown bp_category_encoded value: {encoded}")
        return _mapping[encoded]
