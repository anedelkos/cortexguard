"""Factory helpers for selecting an LLM planner backend at runtime."""

from __future__ import annotations

import logging
import os

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
