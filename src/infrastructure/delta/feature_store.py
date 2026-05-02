"""
DeltaFeatureStore — implements IFeatureStore port.

Reads engineered features from the Silver Delta Lake table.
Used by both AuditService and InferenceService as the single source of truth (DRY).
Falls back to None if Silver layer doesn't have the patient yet.
"""
from __future__ import annotations

import logging
import os
from uuid import UUID

logger = logging.getLogger(__name__)

SILVER_PATH = os.getenv("DELTA_LAKE_PATH", "./data/lakehouse") + "/silver/cardio_features"


class DeltaFeatureStore:
    """
    Implements IFeatureStore.

    For thesis scale (70K patients), we use pyarrow filtering
    without Spark. For production scale, replace with DeltaRS queries.
    """

    async def get_features(self, patient_id: UUID) -> dict[str, float] | None:
        """
        Retrieve the latest Silver layer features for a patient.
        Returns None if patient is not in the feature store yet.
        """
        try:
            import pyarrow.dataset as ds
            dataset = ds.dataset(SILVER_PATH, format="parquet")
            table = dataset.to_table(
                filter=ds.field("patient_id") == str(patient_id)
            )
            if len(table) == 0:
                return None

            row = table.to_pydict()
            return {
                "age_years":              float(row["age_years"][-1]),
                "gender":                 float(row["gender"][-1]),
                "height_cm":              float(row["height_cm"][-1]),
                "weight_kg":              float(row["weight_kg"][-1]),
                "ap_hi":                  float(row["ap_hi"][-1]),
                "ap_lo":                  float(row["ap_lo"][-1]),
                "cholesterol":            float(row["cholesterol"][-1]),
                "glucose":                float(row["glucose"][-1]),
                "is_smoker":              float(row["is_smoker"][-1]),
                "drinks_alcohol":         float(row["drinks_alcohol"][-1]),
                "is_physically_active":   float(row["is_physically_active"][-1]),
                "bmi":                    float(row["bmi"][-1]),
                "pulse_pressure":         float(row["pulse_pressure"][-1]),
                "mean_arterial_pressure": float(row["mean_arterial_pressure"][-1]),
                "bp_category_encoded":    float(row["bp_category_encoded"][-1]),
            }
        except Exception as exc:
            logger.debug("FeatureStore get_features failed for %s: %s", patient_id, exc)
            return None

    async def get_features_bulk(
        self, patient_ids: list[UUID]
    ) -> dict[UUID, dict[str, float]]:
        results = {}
        for pid in patient_ids:
            features = await self.get_features(pid)
            if features is not None:
                results[pid] = features
        return results
