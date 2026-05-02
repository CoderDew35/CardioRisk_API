# Train LightGBM — Model Training Pipeline

> **Command:** `make train`
> **Runs:** `uv run python ml/pipelines/04_train_lgbm.py`

## Purpose

The training pipeline builds the core cardiovascular disease prediction model. It loads patient records from PostgreSQL, engineers features, runs Optuna Bayesian hyperparameter optimization with 50 trials, evaluates the final model against baselines, and registers the model in the MLflow registry.

This script is also **reused by the ContinuousTrainingService** (with 20 trials) for autonomous retraining after drift detection.

## How It Works

```mermaid
flowchart TD
    A["PostgreSQL<br/>patient_cardiovascular_records"] --> B["Load Data<br/>~68K records"]
    B --> C["Feature Engineering<br/>pulse_pressure, MAP"]
    C --> D["Stratified Split<br/>70/15/15"]

    D --> E["Baseline Models"]
    E --> E1["LogisticRegression<br/>AUC ≈ 0.79"]
    E --> E2["RandomForest<br/>AUC ≈ 0.76"]

    D --> F["Optuna HPO<br/>50 trials"]
    F --> G["LightGBM<br/>Best Params"]
    G --> H["Evaluate on Test Set"]

    H --> I{"AUC ≥ 0.70?"}
    I -->|"yes"| J["Save Artifacts"]
    I -->|"no"| K["❌ Fail<br/>AssertionError"]

    J --> J1["ml/models/lgbm_cardio_v1.joblib"]
    J --> J2["ml/models/shap_explainer_v1.joblib"]
    J --> J3["ml/models/feature_importance.csv"]

    J --> L["MLflow Logging"]
    L --> L1["Log params + metrics"]
    L --> L2["Register model<br/>cardiorisk-lgbm"]
    L --> L3["Promote to Production"]

    style A fill:#0ea5e9,color:#fff
    style F fill:#10b981,color:#fff
    style G fill:#10b981,color:#fff
    style L2 fill:#6366f1,color:#fff
    style L3 fill:#22c55e,color:#fff
```

## Pipeline Stages

```mermaid
sequenceDiagram
    participant DB as PostgreSQL
    participant FE as Feature Engineering
    participant BL as Baselines
    participant HPO as Optuna (50 trials)
    participant EVAL as Evaluation
    participant DISK as ml/models/
    participant MLF as MLflow Registry

    DB->>FE: Load 68K records
    FE->>FE: Add pulse_pressure, MAP
    FE->>BL: Train/Val/Test split (70/15/15)
    BL->>BL: LogReg AUC ≈ 0.79
    BL->>BL: RandomForest AUC ≈ 0.76
    FE->>HPO: X_train, y_train, X_val, y_val
    HPO->>HPO: 50 Bayesian trials
    HPO-->>EVAL: Best LightGBM model
    EVAL->>EVAL: AUC-ROC, AUPRC, Brier Score
    EVAL->>DISK: Save model + SHAP explainer
    EVAL->>MLF: Log run + register model
    MLF->>MLF: Promote v1 to Production
```

## Feature Engineering

15 features are used for training:

| # | Feature | Source | Type |
|---|---------|--------|------|
| 1 | `age_years` | `age_days / 365.25` | Continuous |
| 2 | `gender` | Raw | Binary (1=F, 2=M) |
| 3 | `height_cm` | Raw | Continuous |
| 4 | `weight_kg` | Raw | Continuous |
| 5 | `ap_hi` | Raw | Continuous (systolic BP) |
| 6 | `ap_lo` | Raw | Continuous (diastolic BP) |
| 7 | `cholesterol` | Raw | Ordinal (1-3) |
| 8 | `glucose` | Raw | Ordinal (1-3) |
| 9 | `is_smoker` | Raw | Binary |
| 10 | `drinks_alcohol` | Raw | Binary |
| 11 | `is_physically_active` | Raw | Binary |
| 12 | `bmi` | `weight / (height/100)²` | Derived |
| 13 | `pulse_pressure` | `ap_hi - ap_lo` | Derived |
| 14 | `mean_arterial_pressure` | `ap_lo + pp/3` | Derived |
| 15 | `bp_category_encoded` | BPClassifier | Derived (0-4) |

**Target:** `has_cardiovascular_disease` (binary: 0/1)

## Optuna Search Space

```mermaid
flowchart LR
    subgraph HyperparamSpace["Optuna Search Space"]
        A["n_estimators<br/>100–1000"]
        B["learning_rate<br/>0.005–0.3"]
        C["num_leaves<br/>15–127"]
        D["max_depth<br/>3–8"]
        E["min_child_samples<br/>10–100"]
        F["subsample<br/>0.5–1.0"]
        G["colsample_bytree<br/>0.5–1.0"]
        H["reg_alpha<br/>1e-8–10"]
        I["reg_lambda<br/>1e-8–10"]
    end

    subgraph Strategy["Optimization"]
        J["TPE Sampler<br/>(Bayesian)"]
        K["5-Fold CV<br/>Stratified"]
        L["Objective:<br/>Maximize AUC-ROC"]
    end

    HyperparamSpace --> Strategy
```

## Model Performance (Typical Results)

| Metric | Value | Threshold |
|--------|-------|-----------|
| **AUC-ROC** | 0.8007 | ≥ 0.70 (quality bar) |
| **AUPRC** | 0.7844 | — |
| **Brier Score** | 0.1803 | — |
| **Accuracy** | 0.74 | — |

### Top 10 Features by Gain

| Feature | Importance (Gain) |
|---------|:-:|
| `ap_hi` | ████████████████████ 247,636 |
| `age_years` | █████ 53,712 |
| `cholesterol` | ███ 28,931 |
| `mean_arterial_pressure` | ██ 16,475 |
| `bmi` | █ 10,171 |
| `bp_category_encoded` | █ 7,863 |
| `weight_kg` | █ 7,524 |
| `height_cm` | ▌ 3,680 |
| `glucose` | ▌ 3,487 |
| `is_physically_active` | ▎ 2,634 |

## Output Artifacts

| File | Content |
|------|---------|
| `ml/models/lgbm_cardio_v1.joblib` | Trained LightGBM model (serialized) |
| `ml/models/shap_explainer_v1.joblib` | SHAP TreeExplainer (pre-built for speed) |
| `ml/models/feature_importance.csv` | Feature importance table (gain + split) |
| MLflow Registry | `cardiorisk-lgbm` v1, stage: Production |

## MLflow Integration

```mermaid
flowchart LR
    A["Training Complete"] --> B["mlflow.start_run()"]
    B --> C["Log Params<br/>n_estimators, lr, ..."]
    B --> D["Log Metrics<br/>auc_roc, auprc, brier"]
    B --> E["Log Model<br/>mlflow.lightgbm.log_model()"]
    E --> F["Register as<br/>cardiorisk-lgbm"]
    F --> G["Promote to<br/>Production"]

    style F fill:#6366f1,color:#fff
    style G fill:#22c55e,color:#fff
```

After training, visit **http://localhost:5050** to see:
- Experiment runs with all hyperparameters
- Metrics comparison across runs
- Model artifacts stored in MinIO
- Model registry with version history

## Prerequisites

- `make compose-up` (PostgreSQL + MLflow + MinIO)
- `make seed-db` (data loaded)
- `.env` with `DATABASE_URL`, `MLFLOW_TRACKING_URI`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

## When to Use

- **Once**, for initial model training
- **Again**, if you want to retrain with different parameters
- **Automatically**, called by the ContinuousTrainingService (with `n_trials=20`)
