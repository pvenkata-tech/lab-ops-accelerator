from __future__ import annotations

import json
import logging
from pathlib import Path

import boto3
import psycopg

from lab_ops_guardian.config import get_settings

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS protocols (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    exception_type TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(1024),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS protocols_embedding_idx
    ON protocols USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);
"""


def init_knowledge_base() -> None:
    settings = get_settings()
    with psycopg.connect(settings.checkpoint_database_url) as conn:
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()
    logger.info("Protocol knowledge base initialised")


def seed_protocols(protocols_dir: str = "samples/protocols") -> None:
    settings = get_settings()
    bedrock = boto3.client("bedrock-runtime", region_name=settings.aws_region)
    protocols_path = Path(protocols_dir)
    if not protocols_path.exists():
        logger.warning("Protocols directory not found: %s", protocols_dir)
        return

    with psycopg.connect(settings.checkpoint_database_url) as conn:
        for file in protocols_path.glob("*.json"):
            protocol = json.loads(file.read_text())
            embedding = _embed(bedrock, settings, protocol["content"])
            conn.execute(
                """
                INSERT INTO protocols (id, title, exception_type, content, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                    SET content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        updated_at = NOW()
                """,
                (
                    protocol["id"],
                    protocol["title"],
                    protocol["exception_type"],
                    protocol["content"],
                    embedding,
                ),
            )
        conn.commit()
    logger.info("Protocol knowledge base seeded from %s", protocols_dir)


def _embed(bedrock_client, settings, text: str) -> list[float]:
    response = bedrock_client.invoke_model(
        modelId=settings.bedrock_embedding_model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "inputText": text,
            "dimensions": settings.embedding_dimensions,
            "normalize": True,
        }),
    )
    body = json.loads(response["body"].read())
    return body["embedding"]
