# DriftDetectionService — MLOps Tier

> **Command:** `make drift`
> **Runs:** `uv run python services/drift_service/main.py`

## Purpose

The DriftDetectionService is the system's **data quality sentinel**. It monitors incoming patient data for statistical drift — detecting when the distribution of real-world data diverges from the training data. This is critical for ML model reliability: if patient demographics shift (e.g., hospital sees older or sicker patients), the model's predictions may degrade silently.

When drift is detected, the service autonomously triggers the Continuous Training pipeline.

## How It Works

1. At startup, loads the **first 20,000 training records** from PostgreSQL as the reference distribution
2. Connects to RabbitMQ and listens on queue `drift.telemetry.q`
3. For each `PatientTelemetryReceived` event, extracts 15 clinical features
4. Accumulates features into a **sliding window** (default: 500 records)
5. Every `DRIFT_WINDOW_SIZE` records, runs two statistical tests:
   - **Kolmogorov-Smirnov (KS) test** — per-feature, detects distribution shift
   - **Population Stability Index (PSI)** — measures magnitude of shift
6. If drift criteria are met, publishes `ModelDriftDetected` event → triggers CT pipeline

## Architecture

```mermaid
flowchart TD
    A["POST /v1/patients/ingest"] -->|"publishes event"| B["RabbitMQ<br/>Topic Exchange"]
    B -->|"routing_key:<br/>patient.telemetry.raw"| C["Queue: drift.telemetry.q"]
    C -->|"consumes"| D["DriftDetectionService"]

    D --> E["Extract 15<br/>Clinical Features"]
    E --> F["Sliding Window<br/>(500 records)"]
    F -->|"window full"| G{"Statistical Tests"}

    G -->|"KS p < 0.01<br/>for ≥3 features"| H["DRIFT DETECTED"]
    G -->|"PSI > 0.2<br/>any feature"| H
    G -->|"no drift"| I["Pass ✓<br/>continue monitoring"]

    H -->|"publishes"| J["ModelDriftDetected<br/>Event"]
    J -->|"consumed by"| K["ContinuousTraining<br/>Service"]

    style A fill:#0ea5e9,color:#fff
    style B fill:#f59e0b,color:#fff
    style D fill:#22c55e,color:#fff
    style G fill:#eab308,color:#000
    style H fill:#ef4444,color:#fff
    style I fill:#22c55e,color:#fff
    style K fill:#8b5cf6,color:#fff
```

## Statistical Tests

```mermaid
flowchart LR
    subgraph Reference["Reference Distribution (20K records)"]
        R1["ap_hi: μ=128, σ=17"]
        R2["weight_kg: μ=74, σ=14"]
        R3["age_years: μ=53, σ=7"]
    end

    subgraph Window["Sliding Window (500 records)"]
        W1["ap_hi: μ=148, σ=18"]
        W2["weight_kg: μ=89, σ=15"]
        W3["age_years: μ=53, σ=7"]
    end

    R1 ---|"KS test"| W1
    R2 ---|"KS test"| W2
    R3 ---|"KS test"| W3

    W1 --> D1["p < 0.01 ⚠️ DRIFT"]
    W2 --> D2["p < 0.01 ⚠️ DRIFT"]
    W3 --> D3["p = 0.42 ✓ OK"]

    D1 --> V["≥3 features drifted?"]
    D2 --> V
    D3 --> V

    style D1 fill:#ef4444,color:#fff
    style D2 fill:#ef4444,color:#fff
    style D3 fill:#22c55e,color:#fff
```

### Kolmogorov-Smirnov (KS) Test

The KS test measures the maximum distance between two cumulative distribution functions. It answers: *"Are these two samples drawn from the same distribution?"*

- **H₀**: Reference and window come from the same distribution
- **Threshold**: `p-value < 0.01` → reject H₀ (drift detected for that feature)
- **Trigger**: Drift must be detected in **≥3 features** simultaneously to avoid false positives

### Population Stability Index (PSI)

PSI quantifies how much a distribution has shifted. It's widely used in credit scoring and clinical ML.

| PSI Range | Interpretation |
|-----------|---------------|
| `< 0.1` | No significant shift |
| `0.1 – 0.2` | Moderate shift — monitor |
| `> 0.2` | Significant shift — **trigger retrain** |

## Monitored Features (15)

| # | Feature | Type | Drift-Sensitive |
|---|---------|------|:---:|
| 1 | `age_years` | Continuous | ⚠️ |
| 2 | `gender` | Binary | |
| 3 | `height_cm` | Continuous | |
| 4 | `weight_kg` | Continuous | ⚠️ |
| 5 | `ap_hi` | Continuous | ⚠️ |
| 6 | `ap_lo` | Continuous | ⚠️ |
| 7 | `cholesterol` | Ordinal (1-3) | |
| 8 | `glucose` | Ordinal (1-3) | |
| 9 | `is_smoker` | Binary | |
| 10 | `drinks_alcohol` | Binary | |
| 11 | `is_physically_active` | Binary | |
| 12 | `bmi` | Derived | ⚠️ |
| 13 | `pulse_pressure` | Derived | ⚠️ |
| 14 | `mean_arterial_pressure` | Derived | ⚠️ |
| 15 | `bp_category_encoded` | Derived | |

> ⚠️ = Features that are most likely to trigger drift in real clinical settings

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `RABBITMQ_URL` | `amqp://guest:guest@localhost:5672/` | RabbitMQ connection string |
| `DATABASE_URL` | — | PostgreSQL connection string |
| `DRIFT_WINDOW_SIZE` | `500` | Records between drift checks |
| `KS_P_THRESHOLD` | `0.01` | KS p-value threshold |
| `KS_MIN_FEATURES` | `3` | Minimum features that must drift |
| `PSI_THRESHOLD` | `0.2` | PSI threshold for significance |

## Thesis Demo Behavior

During `make simulate-stream`, this service will:

1. **Phase 1 (clean)**: Process ~28K clean records → drift checks pass ✓
2. **Phase 2 (drifted)**: Receive records with `ap_hi += 20` and `weight_kg += 15`
3. After accumulating 500 drifted records → **KS test detects divergence** in `ap_hi`, `weight_kg`, `bmi`, `pulse_pressure`, `mean_arterial_pressure` (≥3 features → trigger!)
4. Publishes `ModelDriftDetected` → ContinuousTrainingService takes over

Watch for this in the terminal:
```
DRIFT DETECTED — KS drifted features: 5 (['ap_hi', 'weight_kg', 'bmi', 'pulse_pressure', 'mean_arterial_pressure']), PSI breach: True
Published ModelDriftDetected event_id=abc123...
```
