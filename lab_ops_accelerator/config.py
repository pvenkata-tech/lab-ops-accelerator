from enum import Enum
from functools import lru_cache

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings


class LLMProviderName(str, Enum):
    """Which backend serves the reasoning model. Every provider maps to the same
    unified tool schema, so the orchestration layer never branches on this value —
    it's a config switch, not a code path."""

    BEDROCK = "bedrock"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENAI = "openai"


class Settings(BaseSettings):
    # LLM provider selection — reasoning model backend
    llm_provider: LLMProviderName = LLMProviderName.BEDROCK

    # AWS Bedrock
    aws_region: str = "us-east-1"
    bedrock_claude_model_id: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_dimensions: int = 1024

    # Anthropic — native API (bypasses Bedrock)
    anthropic_api_key: str = ""
    anthropic_model_id: str = "claude-3-5-sonnet-20241022"

    # Google Gemini
    gemini_api_key: str = ""
    gemini_model_id: str = "gemini-1.5-pro"

    # OpenAI
    openai_api_key: str = ""
    openai_model_id: str = "gpt-4o"

    # PostgreSQL
    database_url: str
    checkpoint_database_url: str

    # LIMS / EHR — reached via MCP servers, not called directly by the orchestrator.
    # (LIMS_API_BASE_URL, LIMS_API_KEY, EHR_WEBHOOK_URL, EHR_API_KEY configure those
    # servers themselves; see lab_ops_accelerator/mcp_servers/.)
    lims_mcp_server_url: str
    ehr_mcp_server_url: str

    # Observability (optional)
    langchain_api_key: str = ""
    langchain_tracing_v2: bool = False
    langsmith_project: str = "lab-ops-accelerator"

    # Agent tuning
    hitl_confidence_threshold: float = 0.80

    @field_validator("hitl_confidence_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("hitl_confidence_threshold must be between 0 and 1")
        return v

    @model_validator(mode="after")
    def validate_required_fields(self) -> "Settings":
        required = {
            "database_url": self.database_url,
            "checkpoint_database_url": self.checkpoint_database_url,
            "lims_mcp_server_url": self.lims_mcp_server_url,
            "ehr_mcp_server_url": self.ehr_mcp_server_url,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise RuntimeError(
                f"Missing required configuration: {', '.join(missing)}. "
                "Check your .env file."
            )

        provider_key_field = {
            LLMProviderName.ANTHROPIC: "anthropic_api_key",
            LLMProviderName.GEMINI: "gemini_api_key",
            LLMProviderName.OPENAI: "openai_api_key",
        }.get(self.llm_provider)
        if provider_key_field is not None and not getattr(self, provider_key_field):
            raise RuntimeError(
                f"LLM_PROVIDER={self.llm_provider.value} requires "
                f"{provider_key_field.upper()} to be set. Check your .env file."
            )
        return self

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
