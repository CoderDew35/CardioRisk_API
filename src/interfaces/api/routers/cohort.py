"""Cohort aggregates router — population-level statistics from PostgreSQL."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func as sa_func, case, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.db.database import get_db_session
from src.infrastructure.db.models import PatientRecordORM
from src.interfaces.api.schemas import CohortAggregates

router = APIRouter()


@router.get("/aggregates", response_model=CohortAggregates)
async def get_aggregates(
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    """Population-level cardiovascular statistics from seeded dataset."""
    stmt = select(
        sa_func.count(sa_func.distinct(PatientRecordORM.patient_id)).label("total"),
        sa_func.avg(PatientRecordORM.age_years).label("mean_age"),
        sa_func.avg(PatientRecordORM.bmi).label("mean_bmi"),
        sa_func.avg(PatientRecordORM.ap_hi).label("mean_ap_hi"),
        sa_func.avg(PatientRecordORM.ap_lo).label("mean_ap_lo"),
        sa_func.avg(
            case((PatientRecordORM.is_smoker.is_(True), 1), else_=0)
        ).label("smoker_pct"),
        sa_func.avg(
            case((PatientRecordORM.is_physically_active.is_(True), 1), else_=0)
        ).label("active_pct"),
        sa_func.avg(
            case((PatientRecordORM.has_cardiovascular_disease.is_(True), 1), else_=0)
        ).label("cvd_pct"),
    ).where(PatientRecordORM.data_source == "batch_csv")

    result = await session.execute(stmt)
    row = result.one()

    bp_stmt = (
        select(
            PatientRecordORM.bp_category,
            sa_func.count().label("count"),
        )
        .where(PatientRecordORM.data_source == "batch_csv")
        .group_by(PatientRecordORM.bp_category)
    )
    bp_result = await session.execute(bp_stmt)
    bp_dist = {r.bp_category: r.count for r in bp_result}

    return {
        "total_patients": row.total or 0,
        "mean_age_years": round(float(row.mean_age or 0), 1),
        "mean_bmi": round(float(row.mean_bmi or 0), 1),
        "mean_ap_hi": round(float(row.mean_ap_hi or 0), 1),
        "mean_ap_lo": round(float(row.mean_ap_lo or 0), 1),
        "smoker_pct": round(float(row.smoker_pct or 0) * 100, 1),
        "active_pct": round(float(row.active_pct or 0) * 100, 1),
        "cvd_positive_pct": round(float(row.cvd_pct or 0) * 100, 1),
        "bp_category_distribution": bp_dist,
    }
