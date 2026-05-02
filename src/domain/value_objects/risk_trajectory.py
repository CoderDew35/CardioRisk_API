"""
Domain Value Objects — RiskScore, RiskTrajectoryPoint, SHAPContribution.

Value objects are immutable, validated wrappers around primitive data.
They have no identity — two RiskScore(0.73) instances are identical.
Zero external imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.domain.entities.enums import RiskLevel


#RiskScore ─────────────

@dataclass(frozen=True)
class RiskScore:
    """
    Validated cardiovascular disease risk probability.
    Value is in [0.0, 1.0] — output of the ML model.
    """
    value: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(f"RiskScore must be in [0.0, 1.0], got {self.value}")

    @property
    def percentage(self) -> float:
        return round(self.value * 100, 2)

    @property
    def risk_level(self) -> RiskLevel:
        if self.value < 0.30:
            return RiskLevel.LOW
        elif self.value < 0.60:
            return RiskLevel.MODERATE
        elif self.value < 0.80:
            return RiskLevel.HIGH
        return RiskLevel.VERY_HIGH

    def delta(self, previous: "RiskScore") -> float:
        """Signed change vs a previous score (positive = risk increased)."""
        return round(self.value - previous.value, 4)

    def __repr__(self) -> str:
        return f"RiskScore({self.percentage:.1f}% — {self.risk_level})"


#SHAPContribution ──────

@dataclass(frozen=True)
class SHAPContribution:
    """
    SHAP value for a single feature at a single time step.
    delta_from_previous is None at T=0 (baseline).
    """
    feature_name: str
    feature_value: float
    shap_value: float           # positive = increases risk, negative = decreases
    delta_from_previous: float | None = None  # Δ(SHAP_t - SHAP_{t-1})

    @property
    def direction(self) -> str:
        """Human-readable direction of influence."""
        if self.shap_value > 0:
            return "increases risk"
        elif self.shap_value < 0:
            return "decreases risk"
        return "neutral"

    @property
    def is_worsening(self) -> bool:
        """True if SHAP contribution grew larger (more harmful) since last step."""
        if self.delta_from_previous is None:
            return False
        return self.delta_from_previous > 0


#RiskTrajectoryPoint ───

@dataclass(frozen=True)
class RiskTrajectoryPoint:
    """
    One data point on a patient's temporal risk trajectory.

    time_step:        0 = baseline, 1–5 = synthetic/future months
    risk_score:       model output for this time step
    shap_contributions: ordered list (highest |SHAP| first)
    llm_narrative:    OpenRouter-generated clinical explanation (may be None)
    is_counterfactual: True if this point was produced by a what-if simulation
    """
    time_step: int
    timestamp: datetime
    risk_score: RiskScore
    shap_contributions: tuple[SHAPContribution, ...]
    llm_narrative: str | None = None
    is_counterfactual: bool = False
    counterfactual_label: str | None = None   # e.g. "Stop Smoking", "Reduce BP"

    def top_contributors(self, n: int = 5) -> tuple[SHAPContribution, ...]:
        """Return top-N features by absolute SHAP value."""
        return tuple(
            sorted(self.shap_contributions, key=lambda c: abs(c.shap_value), reverse=True)[:n]
        )

    def to_dict(self) -> dict:
        """Serialise for API response / Delta Gold storage."""
        return {
            "time_step": self.time_step,
            "timestamp": self.timestamp.isoformat(),
            "risk_score": self.risk_score.value,
            "risk_percentage": self.risk_score.percentage,
            "risk_level": self.risk_score.risk_level.value,
            "shap_contributions": [
                {
                    "feature": c.feature_name,
                    "value": c.feature_value,
                    "shap": c.shap_value,
                    "delta": c.delta_from_previous,
                    "direction": c.direction,
                }
                for c in self.shap_contributions
            ],
            "llm_narrative": self.llm_narrative,
            "is_counterfactual": self.is_counterfactual,
            "counterfactual_label": self.counterfactual_label,
        }
