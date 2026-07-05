from __future__ import annotations

import logging

import psycopg

from lab_ops_accelerator.config import get_settings
from lab_ops_accelerator.rag.embeddings import get_embedding_client

logger = logging.getLogger(__name__)


def retrieve_protocol(query: str) -> dict:
    """Retrieve the most relevant handling protocol for a given query."""
    settings = get_settings()
    query_embedding = get_embedding_client(settings).embed(query)

    with psycopg.connect(settings.checkpoint_database_url, connect_timeout=5) as conn:
        row = conn.execute(
            """
            SELECT id, title, content,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM protocols
            ORDER BY embedding <=> %s::vector
            LIMIT 1
            """,
            (query_embedding, query_embedding),
        ).fetchone()

    if row is None:
        logger.warning("No protocol found for query: %s", query[:80])
        return {
            "protocol_id": "UNKNOWN",
            "protocol_text": "No matching protocol found. Escalate for manual review.",
        }

    protocol_id, title, content, similarity = row
    logger.debug("Retrieved protocol %s (similarity=%.3f)", protocol_id, similarity)
    return {
        "protocol_id": protocol_id,
        "protocol_text": f"{title}\n\n{content}",
    }
