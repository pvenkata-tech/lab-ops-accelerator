from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import boto3

from lab_ops_accelerator.config import LLMProviderName, Settings


@dataclass
class LLMResponse:
    text: str
    model_id: str
    input_tokens: int
    output_tokens: int


class LLMClient(Protocol):
    model_id: str

    def invoke(self, system_prompt: str, user_message: str, max_tokens: int) -> LLMResponse: ...


class BedrockClaudeClient:
    """Claude served through AWS Bedrock — the default production path."""

    def __init__(self, model_id: str, region_name: str):
        self.model_id = model_id
        self._client = boto3.client("bedrock-runtime", region_name=region_name)

    def invoke(self, system_prompt: str, user_message: str, max_tokens: int) -> LLMResponse:
        response = self._client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_message}],
            }),
        )
        body = json.loads(response["body"].read())
        usage = body.get("usage", {})
        return LLMResponse(
            text=body["content"][0]["text"].strip(),
            model_id=self.model_id,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )


class AnthropicClient:
    """Claude served through Anthropic's native API (bypasses Bedrock)."""

    def __init__(self, model_id: str, api_key: str):
        import anthropic

        self.model_id = model_id
        self._client = anthropic.Anthropic(api_key=api_key)

    def invoke(self, system_prompt: str, user_message: str, max_tokens: int) -> LLMResponse:
        response = self._client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        # Models with extended thinking enabled emit a ThinkingBlock before the
        # TextBlock, so content[0] isn't reliably the text — find the text block.
        text_block = next(block for block in response.content if block.type == "text")
        return LLMResponse(
            text=text_block.text.strip(),
            model_id=self.model_id,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class GeminiClient:
    """Google Gemini via the native Generative Language API."""

    def __init__(self, model_id: str, api_key: str):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model_id = model_id
        self._genai = genai

    def invoke(self, system_prompt: str, user_message: str, max_tokens: int) -> LLMResponse:
        model = self._genai.GenerativeModel(self.model_id, system_instruction=system_prompt)
        response = model.generate_content(
            user_message,
            generation_config={"max_output_tokens": max_tokens},
        )
        usage = response.usage_metadata
        return LLMResponse(
            text=response.text.strip(),
            model_id=self.model_id,
            input_tokens=getattr(usage, "prompt_token_count", 0),
            output_tokens=getattr(usage, "candidates_token_count", 0),
        )


class OpenAIClient:
    """OpenAI's Chat Completions API."""

    def __init__(self, model_id: str, api_key: str):
        import openai

        self.model_id = model_id
        self._client = openai.OpenAI(api_key=api_key)

    def invoke(self, system_prompt: str, user_message: str, max_tokens: int) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content.strip(),
            model_id=self.model_id,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
        )


def get_llm_client(settings: Settings) -> LLMClient:
    """Resolve the active LLM client from `LLM_PROVIDER`.

    Every provider implements the same `invoke(system_prompt, user_message, max_tokens)`
    interface, so nodes never branch on which model is serving traffic — swapping
    providers is a config change, not a code change.
    """
    provider = settings.llm_provider
    if provider == LLMProviderName.BEDROCK:
        return BedrockClaudeClient(settings.bedrock_claude_model_id, settings.aws_region)
    if provider == LLMProviderName.ANTHROPIC:
        return AnthropicClient(settings.anthropic_model_id, settings.anthropic_api_key)
    if provider == LLMProviderName.GEMINI:
        return GeminiClient(settings.gemini_model_id, settings.gemini_api_key)
    if provider == LLMProviderName.OPENAI:
        return OpenAIClient(settings.openai_model_id, settings.openai_api_key)
    raise ValueError(f"Unknown LLM provider: {provider}")
