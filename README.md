# CardioRisk XAI

**Interactive Explainable AI for Personalized Cardiovascular Risk Trajectories**

An event-driven backend that combines **LightGBM** predictions with **temporal SHAP** explanations and **GenAI clinical narratives** to make cardiovascular risk assessable, transparent, and actionable — for both physicians and patients.

---

## What It Does

1. **Ingests** patient vitals via REST API
2. **Predicts** cardiovascular disease risk using a LightGBM model trained on 68K+ records
3. **Explains** the prediction with SHAP feature attributions
4. **Simulates** risk trajectories over time via Monte Carlo perturbation (temporal SHAP)
5. **Generates** plain-language clinical narratives through OpenRouter (nvidia/nemotron)
6. **Streams** real-time risk alerts to dashboards via WebSocket + RabbitMQ
7. **Detects** data drift using Kolmogorov-Smirnov and PSI statistical tests
8. **Retrains** autonomously when drift is detected — champion/challenger comparison via MLflow
9. **Hot-swaps** the live model without API restart

---

## Tech Stack

| Layer | Technology |
|-------|-----------:|
| API | FastAPI, Pydantic v2, WebSocket |
| ML | LightGBM, SHAP, scikit-learn, Optuna |
| MLOps | MLflow (model registry), scipy (drift testing) |
| LLM | OpenRouter (nvidia/nemotron) |
| Database | PostgreSQL 16 (async via SQLAlchemy + asyncpg) |
| Messaging | RabbitMQ 3.13 (topic exchange, DLQ, at-least-once delivery) |
| Storage | Delta Lake on MinIO (immutable audit logs) |
| Runtime | Python 3.11, uv, Docker Compose |

---

## Prerequisites

- **Python 3.11+** (managed by `uv`)
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — fast Python package manager
- **Docker** and **Docker Compose** — for PostgreSQL, RabbitMQ, MinIO, MLflow
- **OpenRouter API key** — for LLM narratives ([get one here](https://openrouter.ai/keys))

---

## Quick Start

### 1. Clone and set up the environment

```bash
git clone https://github.com/your-org/cardioriskapi.git
cd cardioriskapi

# Create .env from template
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY

# Install Python 3.11 + create venv + install all dependencies
make setup
```

This runs:
```bash
uv venv --python 3.11              # creates .venv/ with Python 3.11
uv pip install -e ".[messaging,delta,db,ml,xai,api,llm,dev]"
uv lock                            # generates uv.lock for reproducibility
```

### 2. Start infrastructure services

```bash
make compose-up
```

This starts the following Docker containers:

| Service | Port | Credentials |
|---------|------|-------------|
| PostgreSQL | `localhost:5432` | `cardiorisk_user` / `cardiorisk_pass` |
| RabbitMQ | `localhost:5672` (AMQP), `localhost:15672` (UI) | `guest` / `guest` |
| MinIO | `localhost:9000` (API), `localhost:9001` (Console) | `minioadmin` / `minioadmin` |
| MLflow | `localhost:5050` (UI) | — |

### 3. Seed the database

```bash
make seed-db
```

Loads the cardiovascular dataset (~68K records) from `ml/data/cardio_dataset.csv` into PostgreSQL.

### 4. Train the model

```bash
make train
```

Runs the LightGBM training pipeline with Optuna hyperparameter optimization:
- Saves the model to `ml/models/lgbm_cardio_v1.joblib`
- Logs params, metrics, and model artifact to **MLflow**
- Registers the model as `cardiorisk-lgbm` v1 in the MLflow Model Registry (stage: `Production`)

Verify in the MLflow UI at **http://localhost:5050**.

### 5. Generate SHAP explainer

```bash
make shap
```

Creates and saves the SHAP TreeExplainer to `ml/models/shap_explainer_v1.joblib`.

### 6. Start the API

```bash
make dev
```

The API server starts at **http://localhost:8000**. Open **http://localhost:8000/docs** for the interactive OpenAPI documentation.

### 7. Start background workers

In separate terminals:

```bash
make audit       # AuditService — writes to Delta Lake
make inference   # InferenceService — runs ML pipeline on new records
make drift       # DriftDetectionService — monitors for data drift
make ct          # ContinuousTrainingService — retrains when drift detected
```

### 8. (Optional) Run the thesis demo

The thesis demo script simulates the full autonomous CT cycle:

```bash
make simulate-stream
```

This runs a 4-phase simulation:

| Phase | What happens |
|-------|-------------|
| **1. Clean stream** | Records 20K→48K streamed via `/ingest`. No drift. |
| **2. Drift injection** | Records 48K→68K with `ap_hi += 20`, `weight_kg += 15`. |
| **3. Drift detection** | DriftDetectionService detects KS divergence, publishes `ModelDriftDetected`. |
| **4. Auto retrain** | CTService retrains challenger, compares vs champion, promotes if better, API hot-swaps. |

> **Prerequisites:** All 4 workers must be running (`make dev`, `make inference`, `make drift`, `make ct`).

Monitor the cycle:
- **DriftService terminal** → watch for `DRIFT DETECTED`
- **CTService terminal** → watch for `PROMOTED` or `NOT PROMOTED`
- **MLflow UI** → http://localhost:5050 (model versions + metrics comparison)
- **API status** → `curl http://localhost:8000/v1/mlops/status`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/v1/patients` | Paginated patient list |
| `GET` | `/v1/patients/{id}` | Patient detail (read-only) |
| `POST` | `/v1/patients/ingest` | Ingest new patient record |
| `GET` | `/v1/patients/{id}/risk` | Risk score + LLM narrative |
| `GET` | `/v1/patients/{id}/shap` | SHAP feature attributions |
| `GET` | `/v1/patients/{id}/trajectory` | Temporal SHAP trajectory |
| `POST` | `/v1/patients/{id}/counterfactual` | What-if simulation |
| `WS` | `/v1/patients/{id}/live` | Real-time risk stream |
| `GET` | `/v1/cohort/aggregates` | Population-level statistics |
| `GET` | `/v1/mlops/status` | Current model version, drift status |
| `GET` | `/v1/mlops/models` | Model registry (all versions + metrics) |

### Example: Ingest a patient

```bash
curl -X POST http://localhost:8000/v1/patients/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "raw_payload": {
      "age": 18393, "gender": 2, "height": 168, "weight": 62,
      "ap_hi": 110, "ap_lo": 80, "cholesterol": 1, "gluc": 1,
      "smoke": 0, "alco": 0, "active": 1, "cardio": 0
    }
  }'
```

```json
{ "patient_id": "d4e5f6a7-b8c9-0123-...", "status": "accepted" }
```

### Example: Get SHAP explanation

```bash
curl http://localhost:8000/v1/patients/{patient_id}/shap
```

```json
{
  "patient_id": "...",
  "risk_score": 0.73,
  "risk_level": "High",
  "shap_contributions": [
    { "feature": "ap_hi", "value": 145.0, "shap": 0.18, "direction": "increases risk" },
    { "feature": "cholesterol", "value": 3.0, "shap": 0.12, "direction": "increases risk" }
  ]
}
```

### Example: Check MLOps status

```bash
curl http://localhost:8000/v1/mlops/status
```

```json
{
  "current_model_version": "v1",
  "model_name": "cardiorisk-lgbm",
  "drift_detected": false,
  "is_training": false
}
```

---

## Project Structure

```
cardioriskapi/
├── src/                          # Application source code
│   ├── domain/                   # Pure domain logic (zero external deps)
│   │   ├── entities/             # PatientCardiovascularRecord, enums
│   │   ├── events/               # Domain events (RiskScoreGenerated, ModelDriftDetected, etc.)
│   │   ├── services/             # BPClassifier, FeatureValidator
│   │   └── value_objects/        # RiskScore, SHAPContribution, RiskTrajectoryPoint
│   ├── application/              # Use cases + port interfaces
│   │   ├── ports/                # Protocol-based interfaces (IPatientRepository, etc.)
│   │   └── use_cases/            # IngestPatientTelemetry, CalculateRiskProfile,
│   │                             # GenerateTemporalSHAP, RunCounterfactual
│   ├── infrastructure/           # External adapters (implements ports)
│   │   ├── db/                   # PostgreSQL: ORM models, repository, session factory
│   │   ├── delta/                # Delta Lake feature store
│   │   ├── llm/                  # OpenRouter LLM gateway
│   │   ├── messaging/            # RabbitMQ publisher, consumer, WebSocket manager
│   │   └── ml/                   # LightGBM adapter (hot-swap), SHAP adapter
│   └── interfaces/               # HTTP layer
│       └── api/
│           ├── main.py           # FastAPI app, lifespan, CORS, hot-swap consumer
│           ├── schemas.py        # Pydantic request/response models
│           ├── dependencies.py   # Adapter singletons, DI factories
│           └── routers/          # patients.py, cohort.py, health.py, mlops.py
├── services/                     # Background workers (RabbitMQ consumers)
│   ├── audit_service/            # Delta Lake audit logger
│   ├── inference_service/        # ML inference worker (LightGBM + SHAP + LLM)
│   ├── drift_service/            # Data drift detection (KS test + PSI)
│   └── ct_service/               # Continuous training (retrain → compare → promote)
├── ml/                           # ML pipelines and explainability
│   ├── data/                     # CSV dataset
│   ├── models/                   # Trained model + SHAP explainer (.joblib)
│   ├── pipelines/                # 00_seed, 04_train, 06_shap, 08_simulate_stream
│   └── explainability/           # Temporal SHAP, cohort SHAP, waterfall builder
├── tests/
│   ├── unit/                     # Domain + perturbation tests
│   ├── integration/              # DB + messaging tests
│   └── e2e/                      # Full API tests
├── docker-compose.yml            # PostgreSQL, RabbitMQ, MinIO, MLflow
├── pyproject.toml                # Dependencies, linting, testing config
├── uv.lock                       # Reproducible dependency lock
├── .python-version               # 3.11 (used by uv)
├── .env.example                  # Environment variables template
├── Makefile                      # All dev commands
└── main.py                       # CLI entry point
```

---

## Architecture

```
┌──────────────┐     POST /ingest     ┌──────────────────┐
│   React UI   │ ──────────────────►  │   FastAPI (API)   │
│  (frontend)  │ ◄─── WebSocket ───── │   port 8000       │
└──────────────┘                      └────────┬─────────┘
                                               │ publish
                                      ┌────────▼─────────┐
                                      │    RabbitMQ       │
                                      │  Topic Exchange   │
                                      └──┬────┬─────┬────┘
                           ┌─────────────┘    │     └──────────────┐
                           ▼                  ▼                    ▼
                  ┌─────────────────┐ ┌───────────────┐  ┌──────────────────┐
                  │  AuditService   │ │  Inference    │  │ DriftDetection   │
                  │  (Delta Lake)   │ │  Service      │  │ Service (KS/PSI) │
                  └────────┬────────┘ └──────┬────────┘  └────────┬─────────┘
                           │                 │                    │
                           ▼                 ▼                    ▼
                  ┌─────────────────┐ ┌───────────────┐  ┌──────────────────┐
                  │  MinIO (S3)     │ │  Dashboard    │  │ ContinuousTraining│
                  │  Bronze Delta   │ │  WebSocket    │  │ Service (MLflow) │
                  └─────────────────┘ │  relay        │  │ Retrain → Compare│
                                      └───────────────┘  │ → Promote → Swap │
                                                         └──────────────────┘
                                                                 │
                                                         ┌───────▼──────────┐
                                                         │  MLflow Registry │
                                                         │  port 5050       │
                                                         └──────────────────┘
```

### MLOps Continuous Training Cycle

```
 Data ingested → DriftService accumulates sliding window
                      │
                 KS p < 0.01 for ≥3 features OR PSI > 0.2?
                      │ yes
                      ▼
              ModelDriftDetected event → RabbitMQ
                      │
                      ▼
              CTService: retrain (Optuna, 20 trials)
                      │
              Challenger vs Champion (AUC-ROC)
                      │
               ┌──────┴──────┐
               │ Better?     │
          yes  │             │  no
               ▼             ▼
        Promote to      Archive challenger
        Production
               │
               ▼
        ModelRetrained event → RabbitMQ
               │
               ▼
        API hot-swap: LightGBMAdapter.reload()
        (zero downtime, CPython GIL atomic swap)
```

---

## Development

### Run tests

```bash
make test-unit    # fast — domain + perturbation tests only
make test         # full suite with coverage report
```

### Lint and type-check

```bash
make lint         # ruff + mypy
make lint-fix     # auto-fix lint issues
```

### Environment variables

Copy `.env.example` to `.env` and fill in your values:

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | ✅ | PostgreSQL connection string |
| `RABBITMQ_URL` | ✅ | RabbitMQ AMQP connection string |
| `OPENROUTER_API_KEY` | ✅ | OpenRouter API key for LLM narratives |
| `MODEL_PATH` | ✅ | Path to trained LightGBM model |
| `SHAP_EXPLAINER_PATH` | ✅ | Path to saved SHAP explainer |
| `MLFLOW_TRACKING_URI` | ✅ | MLflow tracking server URL (default: `http://localhost:5050`) |
| `MLFLOW_EXPERIMENT_NAME` | ❌ | MLflow experiment name (default: `cardiorisk-lgbm`) |
| `DELTA_LAKE_PATH` | ❌ | Local Delta Lake path (default: `./data/lakehouse`) |
| `MINIO_ENDPOINT` | ❌ | MinIO S3 endpoint (default: `http://localhost:9000`) |

### All make commands

```bash
make help         # Show all available commands
```

| Command | Description |
|---------|-------------|
| `make setup` | Create venv + install all deps |
| `make compose-up` | Start Docker services (Postgres, RabbitMQ, MinIO, MLflow) |
| `make compose-down` | Stop Docker services |
| `make seed-db` | Load CSV dataset into PostgreSQL |
| `make train` | Train LightGBM + register in MLflow |
| `make shap` | Generate SHAP explainer |
| `make dev` | Start FastAPI dev server (hot reload) |
| `make audit` | Start AuditService worker |
| `make inference` | Start InferenceService worker |
| `make drift` | Start DriftDetectionService worker |
| `make ct` | Start ContinuousTrainingService worker |
| `make simulate-stream` | Run thesis demo (streaming + drift + CT cycle) |
| `make test` | Run full test suite with coverage |
| `make test-unit` | Run unit tests only |
| `make lint` | Run ruff + mypy |
| `make clean` | Remove caches |

---

## License

This project is part of a PhD thesis. See LICENSE for details.
