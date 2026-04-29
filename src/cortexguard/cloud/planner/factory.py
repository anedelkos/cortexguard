"""Factory helpers for selecting an LLM planner backend at runtime."""

from __future__ import annotations

import logging
import os

from cortexguard.cloud.planner.explain_client import ExplainClientProtocol
from cortexguard.cloud.planner.llm_client import LLMClientProtocol

logger = logging.getLogger(__name__)

_KEY_VARS: dict[str, str] = {
    "groq": "CLOUD_GROQ_API_KEY",
    "anthropic": "CLOUD_ANTHROPIC_API_KEY",
    "openrouter": "CLOUD_OPENROUTER_API_KEY",
    "grok": "CLOUD_XAI_API_KEY",
}


def get_llm_client(backend: str, **kwargs: object) -> LLMClientProtocol:
    key_var = _KEY_VARS.get(backend)
    if key_var and not os.getenv(key_var) and not kwargs.get("api_key"):
        logger.warning(
            "%s not set — falling back to mock LLM (set %s to enable real inference)",
            key_var,
            key_var,
        )
        from cortexguard.cloud.planner.mock_client import MockLLMClient

        return MockLLMClient()

    if backend == "anthropic":
        from cortexguard.cloud.planner.anthropic_client import AnthropicLLMClient

        return AnthropicLLMClient(api_key=kwargs.get("api_key"))  # type: ignore[arg-type]
    if backend == "openrouter":
        from cortexguard.cloud.planner.openrouter_client import OpenRouterLLMClient

        return OpenRouterLLMClient()
    if backend == "grok":
        from cortexguard.cloud.planner.grok_client import GrokLLMClient

        return GrokLLMClient()
    if backend == "groq":
        from cortexguard.cloud.planner.groq_client import GroqLLMClient

        return GroqLLMClient()
    from cortexguard.cloud.planner.mock_client import MockLLMClient

    return MockLLMClient()


def get_explain_client(config: object) -> ExplainClientProtocol:
    """Return the appropriate :class:`ExplainClientProtocol` for *config*.

    If ``cloud_explain_backend`` is non-empty it takes precedence; otherwise
    the value of ``cloud_llm_backend`` is used as the effective backend.
    """
    from cortexguard.cloud.config import CloudConfig

    if not isinstance(config, CloudConfig):
        from cortexguard.cloud.planner.explain_client import MockExplainClient

        return MockExplainClient()

    effective_backend = config.cloud_explain_backend or config.llm_backend

    if effective_backend in ("mock", ""):
        from cortexguard.cloud.planner.explain_client import MockExplainClient

        return MockExplainClient()

    if effective_backend == "ollama":
        from cortexguard.cloud.planner.explain_client import OpenAICompatibleExplainClient

        return OpenAICompatibleExplainClient(
            base_url=config.cloud_explain_base_url or "http://localhost:11434/v1",
            model=config.cloud_explain_model or "llama3.2",
            api_key="ollama",
        )

    if effective_backend == "anthropic":
        from cortexguard.cloud.planner.explain_client import AnthropicExplainClient

        return AnthropicExplainClient(api_key=config.anthropic_api_key)

    if effective_backend == "groq":
        from cortexguard.cloud.planner.explain_client import GroqExplainClient

        return GroqExplainClient(api_key=os.getenv("CLOUD_GROQ_API_KEY", ""))

    if effective_backend == "openrouter":
        from cortexguard.cloud.planner.explain_client import OpenAICompatibleExplainClient

        return OpenAICompatibleExplainClient(
            base_url="https://openrouter.ai/api/v1",
            model=config.cloud_explain_model or "mistralai/mistral-7b-instruct",
            api_key=os.getenv("CLOUD_OPENROUTER_API_KEY", ""),
        )

    # Default fallback
    from cortexguard.cloud.planner.explain_client import MockExplainClient

    return MockExplainClient()
