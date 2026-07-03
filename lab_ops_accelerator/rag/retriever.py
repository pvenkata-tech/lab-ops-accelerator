from __future__ import annotations

import json
import logging

import boto3
import psycopg

from lab_ops_accelerator.config import get_settings

logger = logging.getLogger(__name__)


def retrieve_protocol(query: str) -> dict:
    """Retrieve the most relevant handling protocol for a given query."""
    settings = get_settings()
    bedrock = boto3.client("bedrock-runtime", region_name=settings.aws_region)

    response = bedrock.invoke_model(
        modelId=settings.bedrock_embedding_model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "inputText": query,
            "dimensions": settings.embedding_dimensions,
            "normalize": True,
        }),
    )
    body = json.loads(response["body"].read())
    query_embedding = body["embedding"]

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
