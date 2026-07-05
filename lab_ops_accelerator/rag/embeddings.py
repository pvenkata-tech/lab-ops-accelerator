from __future__ import annotations

import hashlib
import json
import math
from typing import Protocol

import boto3

from lab_ops_accelerator.config import EmbeddingProviderName, Settings


class EmbeddingClient(Protocol):
    def embed(self, text: str) -> list[float]: ...


class BedrockEmbeddingClient:
    """Titan embeddings served through AWS Bedrock — the production path."""

    def __init__(self, model_id: str, region_name: str, dimensions: int):
        self.model_id = model_id
        self.dimensions = dimensions
        self._client = boto3.client("bedrock-runtime", region_name=region_name)

    def embed(self, text: str) -> list[float]:
        response = self._client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "inputText": text,
                "dimensions": self.dimensions,
                "normalize": True,
            }),
        )
        body = json.loads(response["body"].read())
        return body["embedding"]


class LocalHashEmbeddingClient:
    """Deterministic, dependency-free embedding stub for local development and tests.

    Not semantically meaningful (hashes text into a fixed-size unit vector) — it exists
    so RAG retrieval and the knowledge base can run reproducibly with zero cloud
    credentials. Set EMBEDDING_PROVIDER=bedrock for real semantic retrieval.
    """

    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = []
        for i in range(self.dimensions):
            digest = hashlib.sha256(f"{i}:{text}".encode("utf-8")).digest()
            value = int.from_bytes(digest[:8], "big") / 2**64
            vector.append(value * 2 - 1)
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


def get_embedding_client(settings: Settings) -> EmbeddingClient:
    if settings.embedding_provider == EmbeddingProviderName.LOCAL:
        return LocalHashEmbeddingClient(settings.embedding_dimensions)
    return BedrockEmbeddingClient(
        settings.bedrock_embedding_model_id, settings.aws_region, settings.embedding_dimensions
    )
