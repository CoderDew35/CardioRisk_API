"""
00_seed_postgres.py — Load CSV dataset into PostgreSQL

Reads the confirmed dataset CSV and bulk-inserts all records
into the patient_cardiovascular_records table.

Dataset format (confirmed):
  - Comma-separated
  - Columns: id,age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,
             smoke,alco,active,cardio,age_years,bmi,bp_category,bp_category_encoded
  - bp_category_encoded contains the string label (same as bp_category)
  - weight is already float

Run: make seed-db
  or: python ml/pipelines/00_seed_postgres.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = os.getenv("DATASET_CSV_PATH", "ml/data/cardio_dataset.csv")
BATCH_SIZE = 1000


async def seed(session: AsyncSession) -> None:
    from src.infrastructure.db.models import PatientRecordORM
    from src.infrastructure.db.database import Base, engine
    from src.domain.services.bp_classifier import BPClassifier

    # Create tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Loading dataset from: %s", DATASET_PATH)
    # Dataset is COMMA-separated (confirmed from file inspection)
    df = pd.read_csv(DATASET_PATH, sep=",")
    logger.info("Loaded %d rows, columns: %s", len(df), list(df.columns))

    # Rename raw columns → domain field names
    col_map = {
        "age":    "age_days",
        "height": "height_cm",
        "weight": "weight_kg",
        "gluc":   "glucose",
        "smoke":  "is_smoker",
        "alco":   "drinks_alcohol",
        "active": "is_physically_active",
        "cardio": "has_cardiovascular_disease",
    }
    df = df.rename(columns=col_map)

    # Derive age_years / bmi if missing
    if "age_years" not in df.columns:
        df["age_years"] = (df["age_days"] / 365.25).round(2)
    if "bmi" not in df.columns:
        df["bmi"] = (df["weight_kg"] / ((df["height_cm"] / 100) ** 2)).round(4)

    # bp_category_encoded in this dataset contains the STRING label (not int).
    # We re-derive the integer encoding from ap_hi / ap_lo via BPClassifier.
    def _encode_bp(row) -> int:
        try:
            cat = BPClassifier.classify(int(row["ap_hi"]), int(row["ap_lo"]))
            return BPClassifier.encode(cat)
        except Exception:
            return 2  # Default: Hypertension Stage 1

    df["bp_category_encoded_int"] = df.apply(_encode_bp, axis=1)
    # Ensure bp_category is the string label
    if "bp_category" not in df.columns:
        df["bp_category"] = df["bp_category_encoded_int"].apply(
            lambda e: BPClassifier.from_encoded(e).value
        )

    total = len(df)
    inserted = 0
    skipped = 0

    for start in range(0, total, BATCH_SIZE):
        batch = df.iloc[start: start + BATCH_SIZE]
        orm_records = []

        for _, row in batch.iterrows():
            try:
                # Skip clinically impossible rows (dataset outliers)
                ap_hi, ap_lo = int(row["ap_hi"]), int(row["ap_lo"])
                if ap_lo >= ap_hi or ap_hi > 250 or ap_hi < 60:
                    skipped += 1
                    continue
                height_cm = int(row["height_cm"])
                if height_cm < 100 or height_cm > 250:
                    skipped += 1
                    continue

                orm_records.append(
                    PatientRecordORM(
                        id=uuid.uuid4(),
                        patient_id=uuid.uuid4(),
                        age_days=int(row["age_days"]),
                        gender=int(row["gender"]),
                        height_cm=height_cm,
                        weight_kg=float(row["weight_kg"]),
                        ap_hi=ap_hi,
                        ap_lo=ap_lo,
                        cholesterol=int(row["cholesterol"]),
                        glucose=int(row["glucose"]),
                        is_smoker=bool(int(row["is_smoker"])),
                        drinks_alcohol=bool(int(row["drinks_alcohol"])),
                        is_physically_active=bool(int(row["is_physically_active"])),
                        age_years=float(row["age_years"]),
                        bmi=float(row["bmi"]),
                        bp_category=str(row["bp_category"]),
                        bp_category_encoded=int(row["bp_category_encoded_int"]),
                        has_cardiovascular_disease=bool(int(row["has_cardiovascular_disease"])),
                        recorded_at=datetime.now(timezone.utc),
                        data_source="batch_csv",
                    )
                )
            except Exception as exc:
                logger.warning("Skipping row %s: %s", row.get("id", "?"), exc)
                skipped += 1

        session.add_all(orm_records)
        await session.commit()
        inserted += len(orm_records)
        logger.info("Inserted %d / %d (skipped %d)", inserted, total, skipped)

    logger.info(
        "Seeding complete. Inserted=%d  Skipped=%d  Total=%d",
        inserted, skipped, total,
    )


async def main() -> None:
    from src.infrastructure.db.database import AsyncSessionFactory
    async with AsyncSessionFactory() as session:
        await seed(session)


if __name__ == "__main__":
    asyncio.run(main())
