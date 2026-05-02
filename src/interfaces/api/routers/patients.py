"""
Patient API Router

Endpoints:
  GET    /v1/patients                           → Paginated patient list
  GET    /v1/patients/{id}                      → Patient detail (read-only)
  POST   /v1/patients/ingest                    → Ingest raw patient record
  GET    /v1/patients/{id}/risk                 → Latest risk score
  GET    /v1/patients/{id}/shap                 → Single-step SHAP values
  GET    /v1/patients/{id}/trajectory           → Temporal SHAP trajectory
  POST   /v1/patients/{id}/counterfactual       → What-if simulation
  WS     /v1/patients/{id}/live                 → Real-time risk stream
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.requests import Request

from src.infrastructure.db.patient_repository import PostgreSQLPatientRepository
from src.application.use_cases.ingest_patient_telemetry import (
    IngestPatientTelemetryUseCase,
    ValidationError,
)
from src.application.use_cases.calculate_risk_profile import CalculateRiskProfileUseCase
from src.application.use_cases.generate_temporal_shap import GenerateTemporalSHAPUseCase
from src.application.use_cases.run_counterfactual import RunCounterfactualUseCase

from src.interfaces.api.schemas import (
    CounterfactualRequest,
    CounterfactualResponse,
    IngestRequest,
    IngestResponse,
    PatientListResponse,
    PatientSummary,
    RiskResponse,
    SHAPResponse,
    TrajectoryResponse,
)
from src.interfaces.api.dependencies import (
    explainer,
    feature_store,
    get_repo,
    llm,
    model,
)

router = APIRouter()


# ── Routes ─────────────────────────────────────────────────────────────────

@router.get("", response_model=PatientListResponse)
async def list_patients(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    repo: PostgreSQLPatientRepository = Depends(get_repo),
) -> dict:
    """Paginated patient list (latest record per patient)."""
    rows = await repo.list_patients(offset=offset, limit=limit)
    total = await repo.count_patients()

    patients = [
        PatientSummary(
            patient_id=str(r.patient_id),
            age_years=r.age_years,
            gender=r.gender,
            height_cm=r.height_cm,
            weight_kg=r.weight_kg,
            ap_hi=r.ap_hi,
            ap_lo=r.ap_lo,
            bmi=r.bmi,
            bp_category=r.bp_category,
            cholesterol=r.cholesterol,
            glucose=r.glucose,
            is_smoker=r.is_smoker,
            drinks_alcohol=r.drinks_alcohol,
            is_physically_active=r.is_physically_active,
            has_cardiovascular_disease=r.has_cardiovascular_disease,
        ).model_dump()
        for r in rows
    ]
    return {"patients": patients, "total": total, "offset": offset, "limit": limit}


@router.get("/{patient_id}", response_model=PatientSummary)
async def get_patient_detail(
    patient_id: uuid.UUID,
    repo: PostgreSQLPatientRepository = Depends(get_repo),
) -> dict:
    """Patient demographics and latest vitals (read-only, no ML computation)."""
    record = await repo.get_latest_by_id(patient_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    return PatientSummary(
        patient_id=str(record.patient_id),
        age_years=record.age_years,
        gender=record.gender.value,
        height_cm=record.height_cm,
        weight_kg=record.weight_kg,
        ap_hi=record.ap_hi,
        ap_lo=record.ap_lo,
        bmi=record.bmi,
        bp_category=record.bp_category.value,
        cholesterol=record.cholesterol.value,
        glucose=record.glucose.value,
        is_smoker=record.is_smoker,
        drinks_alcohol=record.drinks_alcohol,
        is_physically_active=record.is_physically_active,
        has_cardiovascular_disease=record.has_cardiovascular_disease,
    ).model_dump()


@router.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest_patient(
    body: IngestRequest,
    request: Request,
    repo: PostgreSQLPatientRepository = Depends(get_repo),
) -> dict:
    """Ingest a raw patient record. Triggers AuditService + InferenceService via RabbitMQ."""
    use_case = IngestPatientTelemetryUseCase(
        patient_repository=repo,
        event_publisher=request.app.state.publisher,
    )
    pid = uuid.UUID(body.patient_id) if body.patient_id else None
    try:
        record = await use_case.execute(body.raw_payload, patient_id=pid, source=body.source)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail={"errors": list(exc.errors)})

    return {"patient_id": str(record.patient_id), "status": "accepted"}


@router.get("/{patient_id}/risk", response_model=RiskResponse)
async def get_latest_risk(
    patient_id: uuid.UUID,
    repo: PostgreSQLPatientRepository = Depends(get_repo),
    request: Request = None,
) -> dict:
    """Return the latest risk score for a patient."""
    use_case = CalculateRiskProfileUseCase(
        patient_repository=repo,
        feature_store=feature_store,
        risk_model=model,
        explainer=explainer,
        llm_gateway=llm,
        event_publisher=request.app.state.publisher,
    )
    try:
        event = await use_case.execute(patient_id=patient_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return {
        "patient_id": str(patient_id),
        "risk_score": event.risk_score,
        "risk_level": event.risk_level,
        "model_version": event.model_version,
        "llm_narrative": event.llm_narrative,
    }


@router.get("/{patient_id}/shap", response_model=SHAPResponse)
async def get_shap_values(
    patient_id: uuid.UUID,
    repo: PostgreSQLPatientRepository = Depends(get_repo),
) -> dict:
    """Single-step SHAP explanation (no trajectory, no LLM). Fast endpoint for waterfall charts."""
    record = await repo.get_latest_by_id(patient_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    features = record.to_feature_dict()
    risk_score = model.predict(features)
    contributions = explainer.explain(features, risk_score)

    return {
        "patient_id": str(patient_id),
        "risk_score": risk_score.value,
        "risk_level": risk_score.risk_level.value,
        "shap_contributions": [
            {
                "feature": c.feature_name,
                "value": c.feature_value,
                "shap": c.shap_value,
                "delta": c.delta_from_previous,
                "direction": c.direction,
            }
            for c in contributions
        ],
    }


@router.get("/{patient_id}/trajectory", response_model=TrajectoryResponse)
async def get_trajectory(
    patient_id: uuid.UUID,
    n_steps: int = 5,
    seed: int | None = None,
    repo: PostgreSQLPatientRepository = Depends(get_repo),
) -> dict:
    """Generate temporal SHAP trajectory (baseline + n_steps Monte Carlo steps)."""
    use_case = GenerateTemporalSHAPUseCase(
        patient_repository=repo,
        feature_store=feature_store,
        risk_model=model,
        explainer=explainer,
        llm_gateway=llm,
    )
    try:
        trajectory = await use_case.execute(
            patient_id=patient_id, n_steps=n_steps, seed=seed
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return {
        "patient_id": str(patient_id),
        "n_steps": n_steps,
        "trajectory": [point.to_dict() for point in trajectory],
    }


@router.post("/{patient_id}/counterfactual", response_model=CounterfactualResponse)
async def run_counterfactual(
    patient_id: uuid.UUID,
    body: CounterfactualRequest,
    repo: PostgreSQLPatientRepository = Depends(get_repo),
) -> dict:
    """What-if simulation: apply feature overrides and return new risk + SHAP."""
    use_case = RunCounterfactualUseCase(
        patient_repository=repo,
        feature_store=feature_store,
        risk_model=model,
        explainer=explainer,
        llm_gateway=llm,
    )
    try:
        result = await use_case.execute(
            patient_id=patient_id,
            feature_overrides=body.feature_overrides,
            include_narrative=body.include_narrative,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return {
        "patient_id": str(patient_id),
        "intervention": result["label"],
        "risk_delta": result["risk_delta"],
        "baseline": result["baseline"].to_dict(),
        "counterfactual": result["counterfactual"].to_dict(),
    }


@router.websocket("/{patient_id}/live")
async def live_risk_stream(websocket: WebSocket, patient_id: uuid.UUID) -> None:
    """WebSocket endpoint: streams RiskScoreGenerated events as they arrive."""
    manager = websocket.app.state.ws_manager
    pid_str = str(patient_id)

    await manager.connect(pid_str, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"status": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(pid_str, websocket)
