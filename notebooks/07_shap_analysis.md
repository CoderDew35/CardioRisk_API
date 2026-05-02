# SHAP Analysis — Explainability Pipeline

> **Command:** `make shap`
> **Runs:** `uv run python ml/pipelines/06_shap_analysis.py`

## Purpose

Generates **global and temporal explainability artifacts** for the trained LightGBM model. This is the thesis's core contribution: making CVD risk predictions transparent for clinicians.

Two types of analysis:
1. **Global cohort SHAP** — which features matter most across the population
2. **Temporal SHAP (Δ-SHAP)** — how each feature's influence changes over simulated time

## Pipeline Flow

```mermaid
flowchart TD
    A["make shap"] --> B["06_shap_analysis.py"]
    B --> C["Step 1: Cohort SHAP<br/>Global feature importance"]
    B --> D["Step 2: Temporal SHAP<br/>Δ-SHAP trajectories"]

    C --> C1["global_shap_summary.csv"]
    D --> D1["sample_trajectories.json"]
    D --> D2["temporal_shap_stats.csv"]

    style B fill:#8b5cf6,color:#fff
    style C fill:#0ea5e9,color:#fff
    style D fill:#10b981,color:#fff
```

## Temporal SHAP — Thesis Novelty

```mermaid
flowchart TD
    A["Patient at t=0"] --> B["t=1: perturb age, weight, BP"]
    B --> C["t=2: perturb again"]
    C --> D["t=3: perturb again"]
    D --> E["t=4: perturb again"]

    A --> S0["SHAP(t=0)"]
    B --> S1["SHAP(t=1)"]
    C --> S2["SHAP(t=2)"]
    D --> S3["SHAP(t=3)"]
    E --> S4["SHAP(t=4)"]

    S0 --> X1["Δ(t=1) = SHAP(t=1) - SHAP(t=0)"]
    S1 --> X1
    S1 --> X2["Δ(t=2) = SHAP(t=2) - SHAP(t=1)"]
    S2 --> X2

    style X1 fill:#f59e0b,color:#fff
    style X2 fill:#f59e0b,color:#fff
```

**Δ-SHAP** shows clinicians *how* feature influence changes: "Your blood pressure's impact on risk is growing — intervene now."

## Output Artifacts

| File | Content |
|------|---------|
| `ml/models/global_shap_summary.csv` | Feature importance rankings |
| `ml/models/sample_trajectories.json` | Temporal SHAP for sample patients |
| `ml/models/temporal_shap_stats.csv` | Aggregated temporal statistics |

## Prerequisites

- `make train` (trained model must exist)
- `make compose-up` (PostgreSQL running)
