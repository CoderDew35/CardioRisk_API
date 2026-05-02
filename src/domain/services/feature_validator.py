"""
FeatureValidator — domain service for clinical data validation.

Guards against dataset outliers and clinically impossible values.
Used by IngestPatientTelemetryUseCase before entity construction.
Zero external dependencies.
"""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @classmethod
    def ok(cls) -> "ValidationResult":
        return cls(is_valid=True, errors=(), warnings=())

    @classmethod
    def failure(cls, *errors: str) -> "ValidationResult":
        return cls(is_valid=False, errors=tuple(errors), warnings=())


class FeatureValidator:
    """
    Validates raw CSV/API payload fields before entity construction.

    Thresholds are based on dataset outlier analysis (EDA Phase):
      - ap_hi: [60, 250] — values outside this range in dataset are errors
      - ap_lo: [40, 200]
      - height: [140, 220] cm (adult range)
      - weight: [30, 200] kg
      - bmi: [10, 60]
    """

    # Clinical plausibility bounds — keys match RAW payload field names
    # (i.e., the CSV/API column names, NOT the domain entity field names).
    # age_years / bmi are derived later by the use case, so not validated here.
    BOUNDS: dict[str, tuple[float, float]] = {
        "age":        (6000,   30000),   # raw column: age in days (~16–82 years)
        "height":     (140,    220),     # raw column: height in cm
        "weight":     (30,     200),     # raw column: weight in kg
        "ap_hi":      (60,     250),
        "ap_lo":      (40,     200),
    }

    CATEGORICAL_BOUNDS: dict[str, tuple[int, int]] = {
        "gender":      (1, 2),
        "cholesterol": (1, 3),
        "gluc":        (1, 3),
        "smoke":       (0, 1),
        "alco":        (0, 1),
        "active":      (0, 1),
    }

    @classmethod
    def validate(cls, payload: dict[str, Any]) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        # Numeric bounds check
        for field, (low, high) in cls.BOUNDS.items():
            if field in payload:
                val = payload[field]
                if val is None:
                    errors.append(f"'{field}' is required but missing")
                elif not (low <= float(val) <= high):
                    errors.append(
                        f"'{field}'={val} is outside plausible range [{low}, {high}]"
                    )

        # Categorical bounds check
        for field, (low, high) in cls.CATEGORICAL_BOUNDS.items():
            if field in payload:
                val = payload[field]
                if val is None:
                    errors.append(f"'{field}' is required")
                elif int(val) not in range(low, high + 1):
                    errors.append(
                        f"'{field}'={val} is invalid; expected value in {list(range(low, high+1))}"
                    )

        # Cross-field validation
        ap_hi = payload.get("ap_hi")
        ap_lo = payload.get("ap_lo")
        if ap_hi is not None and ap_lo is not None:
            if ap_lo >= ap_hi:
                errors.append(
                    f"Diastolic ap_lo={ap_lo} must be less than systolic ap_hi={ap_hi}"
                )

        # Soft warnings (unusual but not impossible)
        if "ap_hi" in payload and payload["ap_hi"] and float(payload["ap_hi"]) > 180:
            warnings.append(
                f"ap_hi={payload['ap_hi']} indicates hypertensive crisis — verify reading"
            )

        if errors:
            return ValidationResult(is_valid=False, errors=tuple(errors), warnings=tuple(warnings))
        return ValidationResult(is_valid=True, errors=(), warnings=tuple(warnings))
