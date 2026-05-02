# InferenceService — Clinical Tier

> **Command:** `make inference`
> **Runs:** `uv run python services/inference_service/main.py`

## Purpose

The InferenceService is the system's **ML inference engine**. When a new patient record arrives, this service runs the complete clinical pipeline: **LightGBM prediction → SHAP explanation → LLM narrative generation → Gold Delta write**. It produces the `RiskScoreGenerated` event that powers real-time WebSocket alerts on the dashboard.

## How It Works

1. Connects to RabbitMQ and listens on queue `inference.raw.q`
2. Receives `PatientTelemetryReceived` events (routing key: `patient.telemetry.raw`)
3. Loads the patient record from PostgreSQL
4. Runs the `CalculateRiskProfileUseCase`:
   - **LightGBM** → predicts cardiovascular disease probability
   - **SHAP** → generates per-feature attribution values
   - **OpenRouter LLM** → generates a plain-language clinical narrative
   - **Delta Lake Gold** → persists enriched risk profile
5. Publishes `RiskScoreGenerated` event → consumed by the API's WebSocket relay for live dashboard updates

## Architecture

```mermaid
flowchart TD
    A["POST /v1/patients/ingest"] -->|"publishes event"| B["RabbitMQ<br/>Topic Exchange"]
    B -->|"routing_key:<br/>patient.telemetry.raw"| C["Queue: inference.raw.q"]
    C -->|"consumes"| D["InferenceService"]

    D --> E["LightGBM<br/>Risk Prediction"]
    D --> F["SHAP<br/>Feature Attribution"]
    D --> G["OpenRouter LLM<br/>Clinical Narrative"]
    D --> H["Delta Lake Gold<br/>Enriched Profiles"]

    D -->|"publishes"| I["RiskScoreGenerated<br/>Event"]
    I -->|"consumed by"| J["API Dashboard<br/>Consumer"]
    J -->|"WebSocket relay"| K["React UI<br/>Live Risk Feed"]

    style A fill:#0ea5e9,color:#fff
    style B fill:#f59e0b,color:#fff
    style D fill:#22c55e,color:#fff
    style E fill:#10b981,color:#fff
    style F fill:#10b981,color:#fff
    style G fill:#8b5cf6,color:#fff
    style H fill:#6366f1,color:#fff
    style I fill:#a855f7,color:#fff
    style K fill:#ec4899,color:#fff
```

## Data Flow Detail

```mermaid
sequenceDiagram
    participant API as FastAPI
    participant RMQ as RabbitMQ
    participant INF as InferenceService
    participant PG as PostgreSQL
    participant ML as LightGBM + SHAP
    participant LLM as OpenRouter
    participant WS as WebSocket Clients

    API->>RMQ: Publish PatientTelemetryReceived
    RMQ->>INF: Deliver to inference.raw.q
    INF->>PG: Load patient record by UUID
    INF->>ML: Predict risk score + SHAP values
    INF->>LLM: Generate clinical narrative
    INF->>PG: Update patient with risk profile
    INF->>RMQ: Publish RiskScoreGenerated
    RMQ->>API: Deliver to dashboard consumer
    API->>WS: Relay to connected clients
```

## ML Pipeline (per message)

```mermaid
flowchart LR
    A["Raw Patient<br/>Record"] --> B["Feature<br/>Engineering"]
    B --> C["LightGBM<br/>predict_proba()"]
    C --> D["Risk Score<br/>0.0 – 1.0"]
    D --> E["Risk Level<br/>Low/Medium/High"]
    B --> F["SHAP<br/>TreeExplainer"]
    F --> G["Feature<br/>Contributions"]
    D --> H["OpenRouter<br/>GPT/Claude"]
    G --> H
    H --> I["Clinical<br/>Narrative"]

    style C fill:#10b981,color:#fff
    style F fill:#0ea5e9,color:#fff
    style H fill:#8b5cf6,color:#fff
```

## Components Used

| Component | Role | Source |
|-----------|------|--------|
| `LightGBMAdapter` | Predicts CVD probability | `ml/models/lgbm_cardio_v1.joblib` |
| `SHAPTreeExplainerAdapter` | Generates per-feature SHAP values | `ml/models/shap_explainer_v1.joblib` |
| `OpenRouterGateway` | Generates clinical narrative via LLM | OpenRouter API (nvidia/nemotron) |
| `PostgreSQLPatientRepository` | Reads/updates patient records | PostgreSQL |
| `DeltaFeatureStore` | Writes enriched profiles to Gold tier | Delta Lake on MinIO |
| `RabbitMQPublisher` | Publishes `RiskScoreGenerated` event | RabbitMQ |

## Key Design Decisions

- **CPU-bound**: Prefetch = 3 (lower than AuditService) because ML inference is computationally expensive
- **Fresh session per message**: Creates a new DB session for each message to prevent stale state
- **Idempotent**: Re-processing the same event produces the same risk score
- **LLM is optional**: If OpenRouter is unavailable, the narrative field will be empty but the risk score still generates

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `RABBITMQ_URL` | `amqp://guest:guest@localhost:5672/` | RabbitMQ connection string |
| `MODEL_PATH` | `ml/models/lgbm_cardio_v1.joblib` | Trained LightGBM model |
| `SHAP_EXPLAINER_PATH` | `ml/models/shap_explainer_v1.joblib` | SHAP TreeExplainer |
| `OPENROUTER_API_KEY` | — | API key for LLM narrative generation |

## When to Use

- **Required for real-time risk scoring.** Without this service, patients are ingested but no risk predictions are generated.
- Must be running if you want live WebSocket risk alerts on the dashboard.
