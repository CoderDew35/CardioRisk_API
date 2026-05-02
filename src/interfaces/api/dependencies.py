"""
Shared FastAPI dependencies for route handlers.

Centralises adapter singletons and repository factory so routers
only declare what they need via Depends().
"""
from __future__ import annotations

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.db.database import get_db_session
from src.infrastructure.db.patient_repository import PostgreSQLPatientRepository
from src.infrastructure.delta.feature_store import DeltaFeatureStore
from src.infrastructure.llm.openrouter_gateway import OpenRouterGateway
from src.infrastructure.ml.lightgbm_adapter import LightGBMAdapter
from src.infrastructure.ml.shap_adapter import SHAPTreeExplainerAdapter


# ── Adapter singletons (initialised once at import time) ─────────────────────

model = LightGBMAdapter()
explainer = SHAPTreeExplainerAdapter(model=model)
llm = OpenRouterGateway()
feature_store = DeltaFeatureStore()


# ── FastAPI dependency callables ─────────────────────────────────────────────

def get_repo(
    session: AsyncSession = Depends(get_db_session),
) -> PostgreSQLPatientRepository:
    """Inject a patient repository scoped to the current DB session."""
    return PostgreSQLPatientRepository(session)
