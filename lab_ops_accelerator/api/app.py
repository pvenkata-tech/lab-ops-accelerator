from __future__ import annotations

import logging

from fastapi import FastAPI

from lab_ops_accelerator.api.routes import router
from lab_ops_accelerator.config import get_settings
from lab_ops_accelerator.rag.knowledge_base import init_knowledge_base, seed_protocols

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Lab Ops Accelerator",
        description="AI agent for specimen exception management and lab throughput optimization",
        version="1.0.0",
    )
    app.include_router(router)

    @app.on_event("startup")
    async def startup():
        logger.info("Starting Lab Ops Accelerator (model=%s)", settings.bedrock_claude_model_id)
        try:
            init_knowledge_base()
            seed_protocols()
        except Exception as exc:
            logger.warning("Knowledge base init failed (will retry on first request): %s", exc)

    return app


app = create_app()
