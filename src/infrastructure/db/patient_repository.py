"""
PostgreSQLPatientRepository — implements IPatientRepository port.

Maps between domain entity (PatientCardiovascularRecord) and ORM model (PatientRecordORM).
All database access is async via SQLAlchemy + asyncpg.
"""
from __future__ import annotations

import uuid
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.entities.enums import BPCategory, CholesterolLevel, Gender, GlucoseLevel
from src.domain.entities.patient_cardiovascular_record import PatientCardiovascularRecord
from src.infrastructure.db.models import PatientRecordORM


class PostgreSQLPatientRepository:
    """Implements IPatientRepository. Injected into use cases via DI."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, record: PatientCardiovascularRecord) -> None:
        orm = self._to_orm(record)
        self._session.add(orm)
        await self._session.flush()

    async def get_by_id(self, patient_id: UUID) -> PatientCardiovascularRecord | None:
        stmt = (
            select(PatientRecordORM)
            .where(PatientRecordORM.patient_id == patient_id)
            .order_by(PatientRecordORM.recorded_at.asc())
        )
        result = await self._session.execute(stmt)
        rows = result.scalars().all()
        if not rows:
            return None
        return self._to_entity(rows[0])

    async def get_latest_by_id(self, patient_id: UUID) -> PatientCardiovascularRecord | None:
        stmt = (
            select(PatientRecordORM)
            .where(PatientRecordORM.patient_id == patient_id)
            .order_by(PatientRecordORM.recorded_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._to_entity(row)

    async def get_trajectory_records(
        self, patient_id: UUID, limit: int = 10
    ) -> list[PatientCardiovascularRecord]:
        stmt = (
            select(PatientRecordORM)
            .where(PatientRecordORM.patient_id == patient_id)
            .order_by(PatientRecordORM.recorded_at.asc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        rows = result.scalars().all()
        return [self._to_entity(r) for r in rows]

    async def list_patients(
        self, offset: int = 0, limit: int = 20
    ) -> list[PatientRecordORM]:
        """Return paginated patient list (latest record per patient_id)."""
        from sqlalchemy import func as sa_func, distinct
        # Get distinct patient_ids with their latest record
        subq = (
            select(
                PatientRecordORM.patient_id,
                sa_func.max(PatientRecordORM.recorded_at).label("max_recorded"),
            )
            .group_by(PatientRecordORM.patient_id)
            .subquery()
        )
        stmt = (
            select(PatientRecordORM)
            .join(
                subq,
                (PatientRecordORM.patient_id == subq.c.patient_id)
                & (PatientRecordORM.recorded_at == subq.c.max_recorded),
            )
            .order_by(PatientRecordORM.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def count_patients(self) -> int:
        from sqlalchemy import func as sa_func
        stmt = select(sa_func.count(sa_func.distinct(PatientRecordORM.patient_id)))
        result = await self._session.execute(stmt)
        return result.scalar_one()

    #Mapping helpers 

    @staticmethod
    def _to_orm(entity: PatientCardiovascularRecord) -> PatientRecordORM:
        from src.domain.services.bp_classifier import BPClassifier
        return PatientRecordORM(
            id=uuid.uuid4(),
            patient_id=entity.patient_id,
            age_days=entity.age_days,
            gender=entity.gender.value,
            height_cm=entity.height_cm,
            weight_kg=entity.weight_kg,
            ap_hi=entity.ap_hi,
            ap_lo=entity.ap_lo,
            cholesterol=entity.cholesterol.value,
            glucose=entity.glucose.value,
            is_smoker=entity.is_smoker,
            drinks_alcohol=entity.drinks_alcohol,
            is_physically_active=entity.is_physically_active,
            age_years=entity.age_years,
            bmi=entity.bmi,
            bp_category=entity.bp_category.value,
            bp_category_encoded=BPClassifier.encode(entity.bp_category),
            has_cardiovascular_disease=entity.has_cardiovascular_disease,
            recorded_at=entity.recorded_at,
            data_source="api",
        )

    @staticmethod
    def _to_entity(orm: PatientRecordORM) -> PatientCardiovascularRecord:
        from src.domain.services.bp_classifier import BPClassifier
        return PatientCardiovascularRecord(
            patient_id=orm.patient_id,
            recorded_at=orm.recorded_at,
            age_days=orm.age_days,
            gender=Gender(orm.gender),
            height_cm=orm.height_cm,
            weight_kg=orm.weight_kg,
            ap_hi=orm.ap_hi,
            ap_lo=orm.ap_lo,
            cholesterol=CholesterolLevel(orm.cholesterol),
            glucose=GlucoseLevel(orm.glucose),
            is_smoker=orm.is_smoker,
            drinks_alcohol=orm.drinks_alcohol,
            is_physically_active=orm.is_physically_active,
            age_years=orm.age_years,
            bmi=orm.bmi,
            bp_category=BPClassifier.from_encoded(orm.bp_category_encoded),
            has_cardiovascular_disease=orm.has_cardiovascular_disease,
        )
