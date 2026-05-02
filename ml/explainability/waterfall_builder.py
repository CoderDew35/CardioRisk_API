"""
waterfall_builder.py

Converts a list of SHAP contribution dicts into chart-ready JSON
for the dashboard waterfall visualization.

Input:  list of {"feature", "value", "shap", "delta"} dicts
        (same shape as RiskTrajectoryPoint.to_dict()["shap_contributions"])
Output: dict with "bars" list, ready for JSON serialisation.
"""
from __future__ import annotations


def build_waterfall(
    contributions: list[dict],
    risk_score: float,
    time_step: int,
) -> dict:
    """
    Args:
        contributions: list of SHAP dicts, each with keys:
                       feature, value, shap, delta
        risk_score:    float in [0, 1]
        time_step:     int (0 = baseline)

    Returns:
        {
          "time_step": int,
          "risk_score": float,
          "risk_pct": float,
          "bars": [
            {"feature": str, "value": float, "shap": float,
             "delta": float|None, "direction": "positive"|"negative"|"neutral"},
            ...
          ]
        }
    """
    bars = []
    for c in contributions:
        shap_val = float(c["shap"])
        if shap_val > 0:
            direction = "positive"
        elif shap_val < 0:
            direction = "negative"
        else:
            direction = "neutral"

        delta = c.get("delta")
        bars.append({
            "feature":   str(c["feature"]),
            "value":     float(c["value"]),
            "shap":      shap_val,
            "delta":     float(delta) if delta is not None else None,
            "direction": direction,
        })

    return {
        "time_step":  int(time_step),
        "risk_score": float(risk_score),
        "risk_pct":   round(float(risk_score) * 100, 2),
        "bars":       bars,
    }
