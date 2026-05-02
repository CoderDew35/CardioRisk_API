"""
CardioRisk XAI API — CLI entry point.

Commands:
  python main.py serve                → Start FastAPI server
  python main.py audit                → Start AuditService worker
  python main.py inference            → Start InferenceService worker
  python main.py seed-db              → Seed PostgreSQL from CSV
"""
from __future__ import annotations

import sys


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "serve"

    if command == "serve":
        import uvicorn
        uvicorn.run(
            "src.interfaces.api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
        )
    elif command == "audit":
        import asyncio
        from services.audit_service.main import main as audit_main
        asyncio.run(audit_main())
    elif command == "inference":
        import asyncio
        from services.inference_service.main import main as inference_main
        asyncio.run(inference_main())
    elif command == "seed-db":
        import asyncio
        import importlib
        mod = importlib.import_module("ml.pipelines.00_seed_postgres")
        asyncio.run(mod.main())
    else:
        print(f"Unknown command: {command}")
        print("Available: serve | audit | inference | seed-db")
        sys.exit(1)


if __name__ == "__main__":
    main()
