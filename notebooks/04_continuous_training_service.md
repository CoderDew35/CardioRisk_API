# ContinuousTrainingService — MLOps Tier

> **Command:** `make ct`
> **Runs:** `uv run python services/ct_service/main.py`

## Purpose

The ContinuousTrainingService is the system's **autonomous model self-healing engine**. When the DriftDetectionService detects that incoming data has diverged from the training distribution, this service automatically retrains a new model, evaluates it against the current champion using a **simulated shadow comparison**, and promotes it to production if it performs better — all without human intervention.

This is the core of the **MLOps Continuous Training (CT)** paradigm in the thesis.

## How It Works

1. Connects to RabbitMQ and listens on queue `ct.drift.q`
2. Receives `ModelDriftDetected` events (routing key: `model.drift.detected`)
3. Queries MLflow registry for the current **champion** model's AUC-ROC
4. Retrains a **challenger** model using Optuna HPO (20 trials, reduced for speed)
5. Compares challenger vs champion on AUC-ROC
6. Decision:
   - **Challenger wins** → Promote to `Production` stage in MLflow
   - **Champion wins** → Archive challenger, keep existing Production model
7. Publishes `ModelRetrained` event → consumed by API for hot-swap

## Architecture

```mermaid
flowchart TD
    A["DriftDetectionService"] -->|"publishes"| B["ModelDriftDetected<br/>Event"]
    B -->|"routing_key:<br/>model.drift.detected"| C["Queue: ct.drift.q"]
    C -->|"consumes"| D["ContinuousTraining<br/>Service"]

    D --> E["Query MLflow Registry<br/>Get Champion AUC-ROC"]
    D --> F["Retrain Challenger<br/>Optuna HPO (20 trials)"]

    E --> G{"Champion vs Challenger<br/>AUC-ROC Comparison"}
    F --> G

    G -->|"challenger > champion"| H["Promote to Production<br/>in MLflow Registry"]
    G -->|"challenger ≤ champion"| I["Archive Challenger<br/>Keep Champion"]

    H --> J["Publish ModelRetrained<br/>(promoted=true)"]
    I --> K["Publish ModelRetrained<br/>(promoted=false)"]

    J -->|"consumed by"| L["API Dashboard Consumer<br/>→ LightGBMAdapter.reload()"]
    L --> M["Zero-Downtime<br/>Model Hot-Swap"]

    style A fill:#ef4444,color:#fff
    style D fill:#22c55e,color:#fff
    style E fill:#6366f1,color:#fff
    style F fill:#10b981,color:#fff
    style H fill:#22c55e,color:#fff
    style I fill:#94a3b8,color:#fff
    style M fill:#0ea5e9,color:#fff
```

## Full CT Cycle — Sequence Diagram

```mermaid
sequenceDiagram
    participant DS as DriftService
    participant RMQ as RabbitMQ
    participant CT as CTService
    participant MLF as MLflow Registry
    participant DB as PostgreSQL
    participant OPT as Optuna HPO
    participant API as FastAPI

    DS->>RMQ: Publish ModelDriftDetected
    RMQ->>CT: Deliver to ct.drift.q
    CT->>MLF: Get champion version + AUC-ROC
    MLF-->>CT: v1, AUC=0.8007

    CT->>DB: Load ALL patient records
    CT->>OPT: Retrain LightGBM (20 trials)
    OPT-->>CT: Best challenger model

    CT->>MLF: Log challenger metrics + model
    MLF-->>CT: Registered as v2

    alt Challenger AUC > Champion AUC
        CT->>MLF: Promote v2 to Production
        CT->>RMQ: Publish ModelRetrained (promoted=true)
        RMQ->>API: Deliver to dashboard consumer
        API->>API: LightGBMAdapter.reload() → hot-swap to v2
    else Champion still better
        CT->>MLF: Archive v2
        CT->>MLF: Re-promote v1 to Production
        CT->>RMQ: Publish ModelRetrained (promoted=false)
    end
```

## Champion vs Challenger Comparison

```mermaid
flowchart LR
    subgraph Champion["Champion (Current Production)"]
        C1["Model v1"]
        C2["AUC-ROC: 0.8007"]
        C3["Trained on: 20K records"]
    end

    subgraph Challenger["Challenger (Retrained)"]
        CH1["Model v2"]
        CH2["AUC-ROC: 0.8089"]
        CH3["Trained on: 68K records<br/>(includes drifted data)"]
    end

    Champion ---|"compare"| Decision{"AUC-ROC<br/>comparison"}
    Challenger ---|"compare"| Decision

    Decision -->|"0.8089 > 0.8007"| Promote["✅ Promote v2<br/>to Production"]

    style Promote fill:#22c55e,color:#fff
    style Decision fill:#eab308,color:#000
```

## Model Hot-Swap Mechanism

When the challenger is promoted, the API receives a `ModelRetrained` event and calls `LightGBMAdapter.reload()`:

```mermaid
flowchart LR
    A["ModelRetrained Event<br/>(promoted=true)"] --> B["API Dashboard<br/>Consumer"]
    B --> C["LightGBMAdapter.reload()"]
    C --> D["Download new model<br/>from MLflow Registry"]
    D --> E["Atomic pointer swap<br/>(CPython GIL)"]
    E --> F["API now serves<br/>with new model v2"]

    style A fill:#a855f7,color:#fff
    style C fill:#0ea5e9,color:#fff
    style E fill:#22c55e,color:#fff
    style F fill:#10b981,color:#fff
```

- **Zero downtime**: The GIL ensures the pointer swap is atomic — in-flight requests complete with the old model, new requests use the new model
- **No restart needed**: The API process stays running
- **Verifiable**: `GET /v1/mlops/status` shows the updated model version

## Key Design Decisions

- **Single-threaded retrain**: `prefetch_count=1` ensures only one retrain runs at a time
- **Guard flag**: `_is_training` boolean prevents duplicate triggers during training
- **Reduced trials**: Uses 20 Optuna trials (vs 50 for initial training) for faster turnaround (~1 minute vs ~15 minutes)
- **Thread pool**: Training runs in `asyncio.run_in_executor()` to avoid blocking the event loop
- **Auto-revert**: If challenger loses, the champion is re-promoted and challenger is archived

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `RABBITMQ_URL` | `amqp://guest:guest@localhost:5672/` | RabbitMQ connection string |
| `DATABASE_URL` | — | PostgreSQL connection string |
| `MLFLOW_TRACKING_URI` | `http://localhost:5050` | MLflow tracking server |
| `MLFLOW_S3_ENDPOINT_URL` | `http://localhost:9000` | MinIO S3 endpoint |
| `AWS_ACCESS_KEY_ID` | `minioadmin` | MinIO access key |
| `AWS_SECRET_ACCESS_KEY` | `minioadmin` | MinIO secret key |
| `CT_N_TRIALS` | `20` | Optuna HPO trials for challenger |

## Thesis Demo Behavior

During `make simulate-stream`, after drift is detected:

1. Receives `ModelDriftDetected` → logs: `CT Cycle triggered — drifted features: ['ap_hi', 'weight_kg', ...]`
2. Queries champion: `Champion: version=1, AUC-ROC=0.8007`
3. Retrains with 20 trials (~1 minute): `Challenger: version=2, AUC-ROC=0.80xx`
4. Comparison result:
   - `PROMOTED: challenger v2 (0.80xx) > champion v1 (0.8007)` **or**
   - `NOT PROMOTED: challenger v2 (0.80xx) <= champion v1 (0.8007)`
5. API hot-swaps (if promoted): `curl localhost:8000/v1/mlops/status` → shows new version

## MLflow Registry View

After the CT cycle, the MLflow UI at http://localhost:5050 will show:

| Version | Stage | AUC-ROC | Trained On |
|---------|-------|---------|-----------|
| v1 | Archived (or Production) | 0.8007 | Initial 20K records |
| v2 | Production (or Archived) | 0.80xx | All 68K+ records |
