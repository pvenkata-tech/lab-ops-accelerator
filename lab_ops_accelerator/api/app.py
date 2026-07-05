from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from lab_ops_accelerator.api.routes import router
from lab_ops_accelerator.config import get_settings
from lab_ops_accelerator.graph.workflow import build_graph
from lab_ops_accelerator.rag.knowledge_base import init_knowledge_base, seed_protocols

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting Lab Ops Accelerator (llm_provider=%s)", settings.llm_provider.value)
    try:
        init_knowledge_base()
        seed_protocols()
    except Exception as exc:
        logger.warning("Knowledge base init failed (will retry on first request): %s", exc)

    async with AsyncPostgresSaver.from_conn_string(settings.checkpoint_database_url) as checkpointer:
        await checkpointer.setup()
        app.state.graph = build_graph(checkpointer)
        yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Lab Ops Accelerator",
        description="AI agent for specimen exception management and lab throughput optimization",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
