# Seed Database — Data Ingestion Pipeline

> **Command:** `make seed-db`
> **Runs:** `uv run python ml/pipelines/00_seed_postgres.py`

## Purpose

The seed pipeline is the **first step** in setting up the CardioRisk system. It loads the raw cardiovascular disease dataset (68K+ records from a Kaggle-derived CSV) into PostgreSQL, transforming raw columns into domain entities ready for ML training and API serving.

This is a **one-time operation** — run it once after `make compose-up` to populate the database.

## How It Works

```mermaid
flowchart TD
    A["ml/data/cardio_dataset.csv<br/>68,205 records"] --> B["Pandas DataFrame"]
    B --> C["Column Renaming<br/>age→age_days, height→height_cm, etc."]
    C --> D["Feature Derivation<br/>age_years, bmi"]
    D --> E["BP Category Encoding<br/>BPClassifier.classify()"]
    E --> F["Clinical Outlier Filter<br/>ap_lo ≥ ap_hi, height < 100"]
    F --> G["Batch Insert<br/>(1000 rows/batch)"]
    G --> H["PostgreSQL<br/>patient_cardiovascular_records"]

    style A fill:#f59e0b,color:#fff
    style H fill:#0ea5e9,color:#fff
```

## Data Transformation Pipeline

```mermaid
flowchart LR
    subgraph Raw["Raw CSV Columns"]
        R1["id"]
        R2["age (days)"]
        R3["gender"]
        R4["height (cm)"]
        R5["weight (kg)"]
        R6["ap_hi"]
        R7["ap_lo"]
        R8["cholesterol"]
        R9["gluc"]
        R10["smoke"]
        R11["alco"]
        R12["active"]
        R13["cardio"]
    end

    subgraph Domain["Domain Entity Fields"]
        D1["patient_id (UUID)"]
        D2["age_days + age_years"]
        D3["gender (enum)"]
        D4["height_cm"]
        D5["weight_kg + bmi"]
        D6["ap_hi + ap_lo"]
        D7["bp_category + bp_category_encoded"]
        D8["cholesterol"]
        D9["glucose"]
        D10["is_smoker"]
        D11["drinks_alcohol"]
        D12["is_physically_active"]
        D13["has_cardiovascular_disease"]
    end

    R1 --> D1
    R2 --> D2
    R3 --> D3
    R4 --> D4
    R5 --> D5
    R6 --> D6
    R7 --> D7
    R8 --> D8
    R9 --> D9
    R10 --> D10
    R11 --> D11
    R12 --> D12
    R13 --> D13
```

## Column Mapping

| CSV Column | Domain Field | Transform |
|-----------|-------------|-----------|
| `id` | `patient_id` | Generate fresh UUID |
| `age` | `age_days`, `age_years` | `age_years = age / 365.25` |
| `gender` | `gender` | 1=Female, 2=Male |
| `height` | `height_cm` | Direct cast |
| `weight` | `weight_kg` | Direct cast to float |
| `ap_hi` | `ap_hi` | Systolic blood pressure |
| `ap_lo` | `ap_lo` | Diastolic blood pressure |
| — | `bp_category` | Derived via `BPClassifier.classify(ap_hi, ap_lo)` |
| — | `bp_category_encoded` | 0=Normal, 1=Elevated, 2=HT1, 3=HT2, 4=Crisis |
| — | `bmi` | `weight_kg / (height_cm / 100)²` |
| `cholesterol` | `cholesterol` | 1=Normal, 2=Above Normal, 3=Well Above |
| `gluc` | `glucose` | 1=Normal, 2=Above Normal, 3=Well Above |
| `smoke` | `is_smoker` | 0/1 → bool |
| `alco` | `drinks_alcohol` | 0/1 → bool |
| `active` | `is_physically_active` | 0/1 → bool |
| `cardio` | `has_cardiovascular_disease` | 0/1 → bool (target variable) |

## Outlier Filtering

Records are skipped if they have clinically impossible values:

| Rule | Rationale |
|------|-----------|
| `ap_lo >= ap_hi` | Diastolic can't exceed systolic |
| `ap_hi > 250` | Exceeds clinical measurement range |
| `ap_hi < 60` | Below viable blood pressure |
| `height_cm < 100` | Below plausible adult height |
| `height_cm > 250` | Above plausible human height |

Typically ~6K records are filtered out, leaving ~62K clean records.

## Output

```
Seeding complete. Inserted=62174  Skipped=6031  Total=68205
```

## Prerequisites

- `make compose-up` (PostgreSQL must be running)
- `.env` configured with `DATABASE_URL`

## When to Use

- **Once**, after initial setup
- Again if you `docker compose down -v` (destroys volumes) and need to re-seed
