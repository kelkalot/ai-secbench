"""
Provider integrations for AI-SecBench.

Supports:
- Anthropic (Claude models)
- OpenAI (GPT models)
- HuggingFace (Inference API and local models)
"""

from typing import Optional, List, Dict, Any

from ai_secbench.providers.base import BaseProvider, ProviderConfig
from ai_secbench.providers.anthropic import AnthropicProvider, create_anthropic_provider
from ai_secbench.providers.openai import OpenAIProvider, create_openai_provider
from ai_secbench.providers.huggingface import (
    HuggingFaceProvider, 
    LocalHuggingFaceProvider,
    create_huggingface_provider,
)
from ai_secbench.providers.xai import XAIProvider, create_xai_provider
from ai_secbench.providers.google import GoogleAIProvider, create_google_provider


# Registry of available providers
PROVIDERS: Dict[str, type] = {
    "anthropic": AnthropicProvider,
    "claude": AnthropicProvider,  # Alias
    "openai": OpenAIProvider,
    "gpt": OpenAIProvider,  # Alias
    "huggingface": HuggingFaceProvider,
    "hf": HuggingFaceProvider,  # Alias
    "huggingface_local": LocalHuggingFaceProvider,
    "xai": XAIProvider,
    "google": GoogleAIProvider,
    "gemini": GoogleAIProvider,  # Alias
}

# Default models for each provider
DEFAULT_MODELS: Dict[str, str] = {
    "anthropic": "claude-3-5-sonnet-20241022",
    "openai": "gpt-4o",
    "huggingface": "meta-llama/Llama-3.1-8B-Instruct",
    "xai": "grok-2-1212",
    "google": "gemini-1.5-pro-latest",
}


def list_providers() -> List[str]:
    """List available provider names."""
    return ["anthropic", "openai", "huggingface", "huggingface_local", "xai", "google"]


def get_provider(
    provider_name: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> BaseProvider:
    """
    Factory function to get a provider instance.
    
    Args:
        provider_name: Name of the provider ("anthropic", "openai", "huggingface")
        model: Model name/ID. Uses default if not specified.
        api_key: API key. Uses environment variable if not specified.
        **kwargs: Additional provider-specific options
    
    Returns:
        Provider instance
    
    Example:
        provider = get_provider("anthropic", model="claude-3-opus-20240229")
        response = await provider.complete([{"role": "user", "content": "Hello!"}])
    """
    provider_name = provider_name.lower()
    
    if provider_name not in PROVIDERS:
        available = ", ".join(list_providers())
        raise ValueError(
            f"Unknown provider: {provider_name}. Available: {available}"
        )
    
    # Get default model if not specified
    if model is None:
        # Map aliases to canonical names for default lookup
        canonical = provider_name
        if provider_name in ("claude", "gpt", "hf"):
            canonical = {"claude": "anthropic", "gpt": "openai", "hf": "huggingface"}[provider_name]
        model = DEFAULT_MODELS.get(canonical, "")
    
    config = ProviderConfig(
        model=model,
        api_key=api_key,
        **kwargs,
    )
    
    provider_class = PROVIDERS[provider_name]
    return provider_class(config)


__all__ = [
    "BaseProvider",
    "ProviderConfig",
    "AnthropicProvider",
    "OpenAIProvider",
    "HuggingFaceProvider",
    "LocalHuggingFaceProvider",
    "XAIProvider",
    "GoogleAIProvider",
    "get_provider",
    "list_providers",
    "create_anthropic_provider",
    "create_openai_provider",
    "create_huggingface_provider",
    "create_xai_provider",
    "create_google_provider",
]
