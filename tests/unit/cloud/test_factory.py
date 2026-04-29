"""Unit tests for the planner factory functions."""

from __future__ import annotations

from cortexguard.cloud.config import CloudConfig
from cortexguard.cloud.planner.explain_client import (
    AnthropicExplainClient,
    GroqExplainClient,
    MockExplainClient,
    OpenAICompatibleExplainClient,
)
from cortexguard.cloud.planner.factory import get_explain_client


def test_get_explain_client_non_config_returns_mock() -> None:
    client = get_explain_client("not-a-config")
    assert isinstance(client, MockExplainClient)


def test_get_explain_client_mock_backend_returns_mock() -> None:
    config = CloudConfig(llm_backend="mock")
    client = get_explain_client(config)
    assert isinstance(client, MockExplainClient)


def test_get_explain_client_empty_explain_backend_falls_back_to_llm_backend() -> None:
    config = CloudConfig(llm_backend="mock", cloud_explain_backend="")
    client = get_explain_client(config)
    assert isinstance(client, MockExplainClient)


def test_get_explain_client_explain_backend_overrides_llm_backend() -> None:
    config = CloudConfig(llm_backend="anthropic", cloud_explain_backend="mock")
    client = get_explain_client(config)
    assert isinstance(client, MockExplainClient)


def test_get_explain_client_ollama_returns_openai_compatible() -> None:
    config = CloudConfig(llm_backend="mock", cloud_explain_backend="ollama")
    client = get_explain_client(config)
    assert isinstance(client, OpenAICompatibleExplainClient)
    assert "11434" in client._base_url


def test_get_explain_client_ollama_uses_custom_url_and_model() -> None:
    config = CloudConfig(
        llm_backend="mock",
        cloud_explain_backend="ollama",
        cloud_explain_base_url="http://custom:11434/v1",
        cloud_explain_model="mistral",
    )
    client = get_explain_client(config)
    assert isinstance(client, OpenAICompatibleExplainClient)
    assert client._base_url == "http://custom:11434/v1"
    assert client._model == "mistral"


def test_get_explain_client_anthropic_returns_anthropic_client() -> None:
    config = CloudConfig(llm_backend="mock", cloud_explain_backend="anthropic")
    client = get_explain_client(config)
    assert isinstance(client, AnthropicExplainClient)


def test_get_explain_client_groq_returns_groq_client() -> None:
    config = CloudConfig(llm_backend="mock", cloud_explain_backend="groq")
    client = get_explain_client(config)
    assert isinstance(client, GroqExplainClient)


def test_get_explain_client_openrouter_returns_openai_compatible() -> None:
    config = CloudConfig(llm_backend="mock", cloud_explain_backend="openrouter")
    client = get_explain_client(config)
    assert isinstance(client, OpenAICompatibleExplainClient)
    assert "openrouter.ai" in client._base_url


def test_get_explain_client_openrouter_uses_custom_model() -> None:
    config = CloudConfig(
        llm_backend="mock",
        cloud_explain_backend="openrouter",
        cloud_explain_model="meta-llama/llama-3-8b",
    )
    client = get_explain_client(config)
    assert isinstance(client, OpenAICompatibleExplainClient)
    assert client._model == "meta-llama/llama-3-8b"


def test_get_explain_client_unknown_backend_returns_mock() -> None:
    config = CloudConfig(llm_backend="mock", cloud_explain_backend="completely_unknown_backend")
    client = get_explain_client(config)
    assert isinstance(client, MockExplainClient)
