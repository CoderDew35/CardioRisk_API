"""
OpenRouter LLM Gateway — implements ILLMGateway port.

Uses the OpenAI-compatible SDK pointed at https://openrouter.ai/api/v1
Model is configurable via OPENROUTER_MODEL env var (default: openai/gpt-4o).

Clinical narrative prompt is structured, templated, and constrained to
prevent hallucination of drug names or treatment recommendations.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from openai import AsyncOpenAI

from src.domain.value_objects.risk_trajectory import RiskScore, SHAPContribution

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "300"))
TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.3"))

SYSTEM_PROMPT = """You are a clinical AI assistant generating patient-friendly cardiovascular risk explanations.

Rules:
- Write in clear, non-alarmist language suitable for patient education
- Keep response to 2-3 sentences maximum
- Do NOT recommend specific medications or treatments
- Do NOT use diagnostic language (do not say "you have" or "you are diagnosed with")
- Focus on which lifestyle factors are contributing most to risk
- If risk improved, acknowledge the positive trend
- Use the patient's actual feature values in your explanation"""


def _build_shap_table(contributions: list[SHAPContribution], top_n: int = 5) -> str:
    """Format top SHAP contributors as a readable table for the LLM prompt."""
    sorted_contribs = sorted(contributions, key=lambda c: abs(c.shap_value), reverse=True)[:top_n]
    lines = ["Feature | Value | SHAP Impact | Trend"]
    lines.append("--------|-------|-------------|------")
    for c in sorted_contribs:
        trend = ""
        if c.delta_from_previous is not None:
            trend = f"{'↑' if c.delta_from_previous > 0 else '↓'}{abs(c.delta_from_previous):.3f}"
        lines.append(
            f"{c.feature_name} | {c.feature_value:.1f} | "
            f"{'+' if c.shap_value > 0 else ''}{c.shap_value:.3f} | {trend}"
        )
    return "\n".join(lines)


class OpenRouterGateway:
    """
    Implements ILLMGateway. Generates clinical narrative from SHAP values.
    Falls back to rule-based template if API is unavailable.
    """

    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )

    async def generate_narrative(
        self,
        patient_context: dict[str, Any],
        shap_contributions: list[SHAPContribution],
        risk_score: RiskScore,
        delta_score: float | None = None,
    ) -> str:
        """
        Generate a 2–3 sentence clinical explanation.

        Args:
            patient_context:    Age, gender, BMI, BP, etc.
            shap_contributions: SHAP values for current prediction
            risk_score:         Current risk score
            delta_score:        Change vs previous score (for counterfactual)

        Returns:
            Clinical narrative string
        """
        shap_table = _build_shap_table(shap_contributions)

        delta_text = ""
        if delta_score is not None:
            direction = "increased" if delta_score > 0 else "decreased"
            delta_text = f"\nRisk has {direction} by {abs(delta_score * 100):.1f} percentage points."

        if "intervention" in patient_context:
            user_prompt = (
                f"Counterfactual simulation: '{patient_context['intervention']}'\n"
                f"Feature changes: {patient_context.get('feature_overrides', {})}\n"
                f"New risk score: {risk_score.percentage:.1f}%{delta_text}\n\n"
                f"Top SHAP contributors after intervention:\n{shap_table}\n\n"
                f"Explain the impact of this intervention in 2-3 sentences."
            )
        else:
            user_prompt = (
                f"Patient profile:\n"
                f"  Age: {patient_context.get('age_years', 'N/A')} years\n"
                f"  Gender: {patient_context.get('gender', 'N/A')}\n"
                f"  BMI: {patient_context.get('bmi', 'N/A'):.1f}\n"
                f"  Blood Pressure: {patient_context.get('ap_hi', 'N/A')}/"
                f"{patient_context.get('ap_lo', 'N/A')} mmHg\n"
                f"  BP Category: {patient_context.get('bp_category', 'N/A')}\n"
                f"  Risk Score: {risk_score.percentage:.1f}% ({risk_score.risk_level}){delta_text}\n\n"
                f"Top risk drivers (SHAP analysis):\n{shap_table}\n\n"
                f"Explain this patient's cardiovascular risk in 2-3 sentences."
            )

        try:
            response = await self._client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            narrative = response.choices[0].message.content or ""
            logger.debug("OpenRouter narrative generated (%d chars)", len(narrative))
            return narrative.strip()

        except Exception as exc:
            logger.warning("OpenRouter API call failed: %s — using rule-based fallback", exc)
            return self._rule_based_fallback(shap_contributions, risk_score)

    @staticmethod
    def _rule_based_fallback(
        contributions: list[SHAPContribution], risk_score: RiskScore
    ) -> str:
        """Template-based fallback when LLM is unavailable."""
        top = sorted(contributions, key=lambda c: abs(c.shap_value), reverse=True)
        top_feature = top[0].feature_name if top else "blood pressure"
        return (
            f"Your current cardiovascular risk score is {risk_score.percentage:.1f}% "
            f"({risk_score.risk_level} risk). "
            f"The most influential factor in your risk assessment is {top_feature}. "
            f"Please consult a healthcare professional for personalised guidance."
        )
