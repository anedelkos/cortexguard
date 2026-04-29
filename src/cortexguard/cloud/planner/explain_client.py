"""Lightweight LLM client for free-text plan explanations."""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class ExplainClientProtocol(Protocol):
    """Minimal protocol for generating plain-English plan explanations."""

    async def explain(self, prompt: str) -> str:
        """Return a plain-English explanation for *prompt*."""
        ...


class MockExplainClient:
    """Returns a fixed mock explanation — no external calls required."""

    async def explain(self, prompt: str) -> str:
        """Return a mock explanation containing the first 100 chars of *prompt*."""
        return f"Mock explanation: {prompt[:100]}"


class OpenAICompatibleExplainClient:
    """Explain client for any OpenAI-compatible REST API (ollama, openrouter, etc.)."""

    def __init__(self, base_url: str, model: str, api_key: str = "ollama") -> None:
        """Initialise with connection parameters."""
        self._base_url = base_url
        self._model = model
        self._api_key = api_key

    async def explain(self, prompt: str) -> str:
        """Call the chat completions endpoint and return the assistant text."""
        from openai import AsyncOpenAI  # type: ignore[import-untyped]

        client: AsyncOpenAI = AsyncOpenAI(base_url=self._base_url, api_key=self._api_key)
        response = await client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        return content if content is not None else ""


class AnthropicExplainClient:
    """Explain client backed by the Anthropic Messages API."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialise with an optional Anthropic API key."""
        self._api_key = api_key or os.getenv("CLOUD_ANTHROPIC_API_KEY", "")

    async def explain(self, prompt: str) -> str:
        """Send *prompt* to Claude and return the plain-text response."""
        import anthropic  # type: ignore[import-untyped]

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        message = await client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in message.content if hasattr(b, "text")]
        return " ".join(text_blocks)


class GroqExplainClient(OpenAICompatibleExplainClient):
    """Explain client backed by the Groq inference API."""

    def __init__(self, api_key: str = "") -> None:
        """Initialise with a Groq API key."""
        super().__init__(
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.1-8b-instant",
            api_key=api_key or os.getenv("CLOUD_GROQ_API_KEY", ""),
        )
