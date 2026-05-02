.PHONY: help setup dev lint test test-unit test-integration compose-up compose-down \
        train shap seed-db clean drift ct simulate-stream

# ─── Help ────────────────
help:
	@echo ""
	@echo "  CardioRisk XAI — Development Commands"
	@echo "  ──────────────────────────────────────"
	@echo "  make setup          Create venv + install all deps (requires uv)"
	@echo "  make compose-up     Start all Docker services"
	@echo "  make compose-down   Stop all Docker services"
	@echo "  make seed-db        Load CSV dataset into PostgreSQL"
	@echo "  make dev            Start FastAPI dev server (hot reload)"
	@echo "  make audit          Start AuditService worker"
	@echo "  make inference      Start InferenceService worker"
	@echo "  make drift          Start DriftDetectionService worker"
	@echo "  make ct             Start ContinuousTrainingService worker"
	@echo "  make lint           Run ruff + mypy"
	@echo "  make test           Run full test suite with coverage"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-int       Run integration tests only"
	@echo "  make train          Run LightGBM training pipeline"
	@echo "  make shap           Run SHAP analysis + save explainer"
	@echo "  make simulate-stream  Run thesis demo (streaming + drift + CT)"
	@echo "  make clean          Remove __pycache__ + .pytest_cache"
	@echo ""

# ─── Setup ───────────────
setup:
	uv venv --python 3.11
	uv pip install -e ".[messaging,delta,db,ml,xai,api,llm,dev]"
	uv lock
	@echo ""
	@echo "  ✅ Setup complete. Activate with: source .venv/bin/activate"
	@echo ""

# ─── Docker ──────────────
compose-up:
	docker compose up -d
	@echo "  RabbitMQ UI  → http://localhost:15672  (guest/guest)"
	@echo "  MinIO Console → http://localhost:9001   (minioadmin/minioadmin)"
	@echo "  PostgreSQL   → localhost:5432           (cardiorisk_user/cardiorisk_pass)"
	@echo "  MLflow UI    → http://localhost:5050"

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f

# ─── Development Server ──
dev:
	uv run uvicorn src.interfaces.api.main:app --reload --host 0.0.0.0 --port 8000

# ─── Services ────────────
audit:
	uv run python services/audit_service/main.py

inference:
	uv run python services/inference_service/main.py

drift:
	uv run python services/drift_service/main.py

ct:
	uv run python services/ct_service/main.py

# ─── Linting ─────────────
lint:
	uv run ruff check src/ services/ ml/ tests/
	uv run mypy src/ services/

lint-fix:
	uv run ruff check --fix src/ services/ ml/ tests/

# ─── Testing ─────────────
test:
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:
	uv run pytest tests/unit/ -v

test-int:
	uv run pytest tests/integration/ -v -m integration

test-e2e:
	uv run pytest tests/e2e/ -v -m e2e

# ─── Data & ML ───────────
seed-db:
	uv run python ml/pipelines/00_seed_postgres.py

train:
	uv run python ml/pipelines/04_train_lgbm.py

shap:
	uv run python ml/pipelines/06_shap_analysis.py

evaluate:
	uv run python ml/pipelines/07_evaluate.py

eda:
	uv run python ml/pipelines/01_eda.py

simulate-stream:
	uv run python ml/pipelines/08_simulate_stream.py

# ─── Migrations ──────────
migrate:
	uv run alembic upgrade head

migration-create:
	uv run alembic revision --autogenerate -m "$(MSG)"

# ─── Clean ───────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete."
