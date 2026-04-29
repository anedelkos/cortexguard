"""Unit tests for explain_client implementations."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cortexguard.cloud.planner.explain_client import (
    AnthropicExplainClient,
    ExplainClientProtocol,
    GroqExplainClient,
    MockExplainClient,
    OpenAICompatibleExplainClient,
)


@pytest.mark.asyncio
async def test_mock_explain_client_returns_prompt_prefix() -> None:
    client = MockExplainClient()
    result = await client.explain("Hello, this is a test prompt.")
    assert result.startswith("Mock explanation:")
    assert "Hello" in result


@pytest.mark.asyncio
async def test_mock_explain_client_truncates_long_prompt() -> None:
    client = MockExplainClient()
    long_prompt = "x" * 200
    result = await client.explain(long_prompt)
    assert len(result) < len(long_prompt) + 30


def test_mock_explain_client_satisfies_protocol() -> None:
    client = MockExplainClient()
    assert isinstance(client, ExplainClientProtocol)


def test_openai_compatible_client_stores_params() -> None:
    client = OpenAICompatibleExplainClient(
        base_url="http://localhost:11434/v1",
        model="llama3.2",
        api_key="test-key",
    )
    assert client._base_url == "http://localhost:11434/v1"
    assert client._model == "llama3.2"
    assert client._api_key == "test-key"


def test_openai_compatible_client_default_api_key() -> None:
    client = OpenAICompatibleExplainClient(base_url="http://localhost/v1", model="m")
    assert client._api_key == "ollama"


@pytest.mark.asyncio
async def test_openai_compatible_client_explain_returns_content() -> None:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Explanation text"

    mock_openai_instance = AsyncMock()
    mock_openai_instance.chat.completions.create = AsyncMock(return_value=mock_response)

    mock_openai_class = MagicMock(return_value=mock_openai_instance)

    client = OpenAICompatibleExplainClient(
        base_url="http://localhost:11434/v1", model="llama3.2", api_key="ollama"
    )
    with patch.dict("sys.modules", {"openai": MagicMock(AsyncOpenAI=mock_openai_class)}):
        result = await client.explain("Explain this plan.")

    assert result == "Explanation text"


@pytest.mark.asyncio
async def test_openai_compatible_client_explain_handles_none_content() -> None:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = None

    mock_openai_instance = AsyncMock()
    mock_openai_instance.chat.completions.create = AsyncMock(return_value=mock_response)

    mock_openai_class = MagicMock(return_value=mock_openai_instance)

    client = OpenAICompatibleExplainClient(
        base_url="http://localhost:11434/v1", model="llama3.2", api_key="ollama"
    )
    with patch.dict("sys.modules", {"openai": MagicMock(AsyncOpenAI=mock_openai_class)}):
        result = await client.explain("Explain this plan.")

    assert result == ""


def test_anthropic_client_stores_api_key() -> None:
    client = AnthropicExplainClient(api_key="test-anthropic-key")
    assert client._api_key == "test-anthropic-key"


def test_anthropic_client_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUD_ANTHROPIC_API_KEY", "env-key")
    client = AnthropicExplainClient()
    assert client._api_key == "env-key"


@pytest.mark.asyncio
async def test_anthropic_client_explain_returns_text() -> None:
    mock_text_block = MagicMock()
    mock_text_block.text = "Anthropic explanation"

    mock_message = MagicMock()
    mock_message.content = [mock_text_block]

    mock_anthropic_instance = AsyncMock()
    mock_anthropic_instance.messages.create = AsyncMock(return_value=mock_message)

    mock_anthropic_class = MagicMock(return_value=mock_anthropic_instance)

    client = AnthropicExplainClient(api_key="test-key")
    with patch.dict("sys.modules", {"anthropic": MagicMock(AsyncAnthropic=mock_anthropic_class)}):
        result = await client.explain("Explain this plan.")

    assert result == "Anthropic explanation"


@pytest.mark.asyncio
async def test_anthropic_client_explain_joins_multiple_blocks() -> None:
    block1 = MagicMock()
    block1.text = "First part"
    block2 = MagicMock()
    block2.text = "Second part"

    mock_message = MagicMock()
    mock_message.content = [block1, block2]

    mock_anthropic_instance = AsyncMock()
    mock_anthropic_instance.messages.create = AsyncMock(return_value=mock_message)

    mock_anthropic_class = MagicMock(return_value=mock_anthropic_instance)

    client = AnthropicExplainClient(api_key="test-key")
    with patch.dict("sys.modules", {"anthropic": MagicMock(AsyncAnthropic=mock_anthropic_class)}):
        result = await client.explain("Explain this.")

    assert "First part" in result
    assert "Second part" in result


def test_groq_explain_client_sets_groq_endpoint() -> None:
    client = GroqExplainClient(api_key="groq-test-key")
    assert "groq.com" in client._base_url
    assert client._api_key == "groq-test-key"


def test_groq_explain_client_default_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUD_GROQ_API_KEY", "env-groq-key")
    client = GroqExplainClient()
    assert client._api_key == "env-groq-key"
