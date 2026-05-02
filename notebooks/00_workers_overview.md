# Background Workers — Overview

This document provides a high-level overview of all 4 background workers in the CardioRisk XAI system. Each worker is an independent RabbitMQ consumer that processes domain events asynchronously.

## System Architecture

```mermaid
flowchart TB
    subgraph Client["Client Layer"]
        UI["React UI<br/>(cardioriskui)"]
    end

    subgraph API["API Layer"]
        FA["FastAPI<br/>port 8000"]
        WS["WebSocket<br/>Manager"]
    end

    subgraph Messaging["Messaging Layer"]
        RMQ["RabbitMQ<br/>Topic Exchange<br/>cardiorisk.events"]
    end

    subgraph Workers["Background Workers"]
        AUD["🔒 AuditService<br/><i>make audit</i>"]
        INF["🧠 InferenceService<br/><i>make inference</i>"]
        DFT["📊 DriftDetectionService<br/><i>make drift</i>"]
        CTS["🔄 ContinuousTrainingService<br/><i>make ct</i>"]
    end

    subgraph Storage["Storage Layer"]
        PG["PostgreSQL"]
        DL["Delta Lake<br/>(MinIO)"]
        MLF["MLflow Registry<br/>port 5050"]
    end

    UI <-->|"REST + WebSocket"| FA
    FA -->|"publish events"| RMQ
    RMQ -->|"patient.telemetry.raw"| AUD
    RMQ -->|"patient.telemetry.raw"| INF
    RMQ -->|"patient.telemetry.raw"| DFT
    RMQ -->|"model.drift.detected"| CTS

    AUD --> DL
    INF --> PG
    INF --> FA
    DFT -->|"drift event"| RMQ
    CTS --> MLF
    CTS -->|"retrained event"| RMQ

    FA --> WS
    WS --> UI

    style AUD fill:#22c55e,color:#fff
    style INF fill:#0ea5e9,color:#fff
    style DFT fill:#f59e0b,color:#fff
    style CTS fill:#8b5cf6,color:#fff
    style RMQ fill:#ef4444,color:#fff
    style MLF fill:#6366f1,color:#fff
```

## Worker Summary

| # | Worker | Command | Queue | Listens For | Publishes | Tier |
|---|--------|---------|-------|-------------|-----------|------|
| 1 | [AuditService](./01_audit_service.md) | `make audit` | `audit.raw.q` | `patient.telemetry.raw` | `AuditLogWritten` | Compliance |
| 2 | [InferenceService](./02_inference_service.md) | `make inference` | `inference.raw.q` | `patient.telemetry.raw` | `RiskScoreGenerated` | Clinical |
| 3 | [DriftDetectionService](./03_drift_detection_service.md) | `make drift` | `drift.telemetry.q` | `patient.telemetry.raw` | `ModelDriftDetected` | MLOps |
| 4 | [ContinuousTrainingService](./04_continuous_training_service.md) | `make ct` | `ct.drift.q` | `model.drift.detected` | `ModelRetrained` | MLOps |

## Event Flow — Complete Chain

```mermaid
flowchart LR
    A["Patient<br/>Ingested"] -->|"event"| B["PatientTelemetry<br/>Received"]
    B --> C["AuditService<br/>→ Delta Lake"]
    B --> D["InferenceService<br/>→ Risk Score"]
    B --> E["DriftService<br/>→ KS/PSI Test"]

    D -->|"event"| F["RiskScoreGenerated"]
    F --> G["WebSocket<br/>→ Live Dashboard"]

    E -->|"drift found"| H["ModelDrift<br/>Detected"]
    H --> I["CTService<br/>→ Retrain"]

    I -->|"event"| J["ModelRetrained"]
    J --> K["API Hot-Swap<br/>→ New Model Live"]

    style A fill:#0ea5e9,color:#fff
    style C fill:#22c55e,color:#fff
    style D fill:#0ea5e9,color:#fff
    style E fill:#f59e0b,color:#fff
    style H fill:#ef4444,color:#fff
    style I fill:#8b5cf6,color:#fff
    style K fill:#10b981,color:#fff
```

## Startup Order

```bash
# 1. Infrastructure (must be first)
make compose-up

# 2. Seed data + train model (one-time)
make seed-db
make train

# 3. API server
make dev

# 4. Workers (any order, separate terminals)
make audit          # Terminal 2
make inference      # Terminal 3
make drift          # Terminal 4
make ct             # Terminal 5

# 5. (Optional) Run thesis demo
make simulate-stream   # Terminal 6
```

## RabbitMQ Topology

```mermaid
flowchart TD
    EX["Exchange: cardiorisk.events<br/>(topic)"]

    EX -->|"patient.telemetry.raw"| Q1["audit.raw.q"]
    EX -->|"patient.telemetry.raw"| Q2["inference.raw.q"]
    EX -->|"patient.telemetry.raw"| Q3["drift.telemetry.q"]
    EX -->|"model.drift.detected"| Q4["ct.drift.q"]
    EX -->|"risk.score.generated"| Q5["dashboard.risk.q"]
    EX -->|"model.retrained"| Q6["dashboard.retrained.q"]

    Q1 --> AUD["AuditService"]
    Q2 --> INF["InferenceService"]
    Q3 --> DFT["DriftDetectionService"]
    Q4 --> CTS["ContinuousTrainingService"]
    Q5 --> DC1["API Dashboard Consumer"]
    Q6 --> DC2["API Dashboard Consumer"]

    style EX fill:#ef4444,color:#fff
    style Q1 fill:#f59e0b,color:#fff
    style Q2 fill:#f59e0b,color:#fff
    style Q3 fill:#f59e0b,color:#fff
    style Q4 fill:#f59e0b,color:#fff
    style Q5 fill:#f59e0b,color:#fff
    style Q6 fill:#f59e0b,color:#fff
```

> **Note:** Each queue has its own dead-letter queue (DLQ) for failed messages. The `BaseRabbitMQConsumer` handles manual acknowledgement, automatic retry, and DLQ routing.

## Common Patterns

All 4 workers share these patterns:

1. **`BaseRabbitMQConsumer`** — Base class providing connection management, manual ACK, DLQ routing, and graceful shutdown
2. **`RabbitMQPublisher`** — Shared publisher for emitting domain events
3. **Graceful shutdown** — `KeyboardInterrupt` triggers clean disconnection from RabbitMQ
4. **Environment-based config** — All settings via `.env` file (loaded with `python-dotenv`)
5. **Structured logging** — Consistent format: `timestamp | level | ServiceName | message`
