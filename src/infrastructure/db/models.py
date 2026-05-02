"""
SQLAlchemy ORM Model for PatientCardiovascularRecord.

Maps the domain entity to a PostgreSQL table.
The dataset CSV will be seeded into this table by ml/pipelines/00_seed_postgres.py.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from src.infrastructure.db.database import Base


class PatientRecordORM(Base):
    __tablename__ = "patient_cardiovascular_records"

    #Primary key ────
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )

    #Dataset columns (raw) ────────────────────────────────────────────────
    age_days: Mapped[int] = mapped_column(Integer, nullable=False)
    gender: Mapped[int] = mapped_column(Integer, nullable=False)       # 1=Male, 2=Female
    height_cm: Mapped[int] = mapped_column(Integer, nullable=False)
    weight_kg: Mapped[float] = mapped_column(Float, nullable=False)
    ap_hi: Mapped[int] = mapped_column(Integer, nullable=False)
    ap_lo: Mapped[int] = mapped_column(Integer, nullable=False)
    cholesterol: Mapped[int] = mapped_column(Integer, nullable=False)  # 1/2/3
    glucose: Mapped[int] = mapped_column(Integer, nullable=False)      # 1/2/3
    is_smoker: Mapped[bool] = mapped_column(Boolean, nullable=False)
    drinks_alcohol: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_physically_active: Mapped[bool] = mapped_column(Boolean, nullable=False)

    #Pre-engineered fields ────────────────────────────────────────────────
    age_years: Mapped[float] = mapped_column(Float, nullable=False)
    bmi: Mapped[float] = mapped_column(Float, nullable=False)
    bp_category: Mapped[str] = mapped_column(String(50), nullable=False)
    bp_category_encoded: Mapped[int] = mapped_column(Integer, nullable=False)

    #Target label (None for live inference records) ───────────────────────
    has_cardiovascular_disease: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    #Metadata ───────
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    data_source: Mapped[str] = mapped_column(
        String(50), nullable=False, default="api"
    )  # "api" | "batch_csv" | "websocket"
