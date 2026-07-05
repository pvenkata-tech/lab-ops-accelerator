from __future__ import annotations

import json
import logging
from pathlib import Path

import psycopg

from lab_ops_accelerator.config import get_settings
from lab_ops_accelerator.rag.embeddings import get_embedding_client

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

-- HNSW rather than IVFFlat: IVFFlat partitions rows into `lists` buckets and by
-- default probes only one of them, so on a knowledge base this small (dozens to a
-- few hundred protocols) a probe can easily land on an empty bucket and silently
-- return zero rows for an ORDER BY ... LIMIT query. HNSW has no such "empty bucket"
-- failure mode and needs no row-count-dependent tuning.
DROP INDEX IF EXISTS protocols_embedding_idx;
CREATE INDEX protocols_embedding_idx
    ON protocols USING hnsw (embedding vector_cosine_ops);
"""


def init_knowledge_base() -> None:
    settings = get_settings()
    with psycopg.connect(settings.checkpoint_database_url) as conn:
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()
    logger.info("Protocol knowledge base initialised")


def seed_protocols(protocols_dir: str = "samples/protocols") -> None:
    settings = get_settings()
    embedding_client = get_embedding_client(settings)
    protocols_path = Path(protocols_dir)
    if not protocols_path.exists():
        logger.warning("Protocols directory not found: %s", protocols_dir)
        return

    with psycopg.connect(settings.checkpoint_database_url) as conn:
        for file in protocols_path.glob("*.json"):
            protocol = json.loads(file.read_text())
            embedding = embedding_client.embed(protocol["content"])
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
