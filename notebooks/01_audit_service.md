# AuditService — Compliance Tier

> **Command:** `make audit`
> **Runs:** `uv run python services/audit_service/main.py`

## Purpose

The AuditService is the system's **immutable compliance record**. It captures every patient record ingested into the platform and writes it to a Bronze-tier Delta Lake table. This data is **never modified or deleted** — it provides a full audit trail for GDPR compliance, clinical traceability, and research reproducibility.

## How It Works

1. Connects to RabbitMQ and listens on queue `audit.raw.q`
2. Receives every `PatientTelemetryReceived` event (routing key: `patient.telemetry.raw`)
3. Extracts the raw payload, patient ID, and event metadata
4. Appends an immutable row to the Bronze Delta Lake table at `./data/lakehouse/bronze/cardio_events`
5. Publishes an `AuditLogWritten` confirmation event back to RabbitMQ

## Architecture

```mermaid
flowchart TD
    A["POST /v1/patients/ingest"] -->|"publishes event"| B["RabbitMQ<br/>Topic Exchange"]
    B -->|"routing_key:<br/>patient.telemetry.raw"| C["Queue: audit.raw.q"]
    C -->|"consumes"| D["AuditService"]
    D -->|"append-only write"| E["Delta Lake Bronze<br/>./data/lakehouse/bronze/cardio_events"]
    D -->|"publishes"| F["AuditLogWritten Event"]

    style A fill:#0ea5e9,color:#fff
    style B fill:#f59e0b,color:#fff
    style C fill:#f59e0b,color:#fff
    style D fill:#22c55e,color:#fff
    style E fill:#6366f1,color:#fff
    style F fill:#a855f7,color:#fff
```

## Data Flow Detail

```mermaid
sequenceDiagram
    participant API as FastAPI
    participant RMQ as RabbitMQ
    participant AUD as AuditService
    participant DL as Delta Lake (MinIO)

    API->>RMQ: Publish PatientTelemetryReceived
    RMQ->>AUD: Deliver to audit.raw.q
    AUD->>AUD: Extract raw_payload, patient_id, event_id
    AUD->>DL: Append row to Bronze table (PyArrow + Delta)
    AUD->>RMQ: Publish AuditLogWritten
    AUD->>RMQ: ACK original message
```

## Bronze Table Schema

| Column | Type | Description |
|--------|------|-------------|
| `patient_id` | string | UUID of the patient |
| `event_id` | string | UUID of the original telemetry event |
| `raw_payload` | string | Full JSON payload (unmodified) |
| `source` | string | Origin: `api`, `batch_csv`, `websocket` |
| `schema_version` | string | Payload schema version (default: `1.0`) |
| `ingested_at` | timestamp (UTC) | When the audit record was created |

## Key Design Decisions

- **Append-only**: Uses `mode="append"` — data is never updated or deleted
- **Schema merge**: New columns are absorbed via `schema_mode="merge"`
- **Raw preservation**: The original JSON payload is stored as-is, not transformed
- **At-least-once**: RabbitMQ manual ACK ensures no records are lost
- **Prefetch 5**: Processes up to 5 messages concurrently for throughput

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `RABBITMQ_URL` | `amqp://guest:guest@localhost:5672/` | RabbitMQ connection string |
| `DELTA_LAKE_PATH` | `./data/lakehouse` | Root path for Delta Lake tables |

## When to Use

- **Always.** This service should run alongside the API in production to maintain a complete audit trail.
- Not required for development/testing if you don't need Delta Lake writes.
