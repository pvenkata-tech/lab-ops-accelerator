from functools import lru_cache

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # AWS Bedrock
    aws_region: str = "us-east-1"
    bedrock_claude_model_id: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_dimensions: int = 1024

    # PostgreSQL
    database_url: str
    checkpoint_database_url: str

    # LIMS integration
    lims_api_base_url: str
    lims_api_key: str

    # EHR / Notification
    ehr_webhook_url: str
    ehr_api_key: str

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
            "lims_api_base_url": self.lims_api_base_url,
            "lims_api_key": self.lims_api_key,
            "ehr_webhook_url": self.ehr_webhook_url,
            "ehr_api_key": self.ehr_api_key,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise RuntimeError(
                f"Missing required configuration: {', '.join(missing)}. "
                "Check your .env file."
            )
        return self

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
