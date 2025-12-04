"""
Provider registry and basic connectivity tests.

These tests avoid real network calls and instead check:
- Registry wiring (list_providers/get_provider)
- Helpful errors when API keys are missing

If provider SDKs are not installed for a given provider, the test is skipped.
"""

import asyncio
import importlib

import pytest

from ai_secbench.providers import get_provider, list_providers


def _is_installed(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def test_registry_contains_expected_providers():
    providers = list_providers()
    for expected in ["anthropic", "openai", "huggingface", "huggingface_local", "xai", "google"]:
        assert expected in providers


@pytest.mark.parametrize(
    ("provider_name", "expected_cls_name"),
    [
        ("anthropic", "AnthropicProvider"),
        ("openai", "OpenAIProvider"),
        ("huggingface", "HuggingFaceProvider"),
        ("huggingface_local", "LocalHuggingFaceProvider"),
        ("xai", "XAIProvider"),
        ("google", "GoogleAIProvider"),
    ],
)
def test_get_provider_returns_correct_class(provider_name, expected_cls_name):
    provider = get_provider(provider_name, model="dummy-model", api_key="DUMMY")
    assert provider.__class__.__name__ == expected_cls_name
    # count_tokens should be callable without network
    assert isinstance(provider.count_tokens("hello world"), int)


@pytest.mark.parametrize(
    ("provider_name", "model", "env_vars", "import_check"),
    [
        ("anthropic", "claude-3-5-sonnet-20241022", ["ANTHROPIC_API_KEY"], "anthropic"),
        ("openai", "gpt-4o", ["OPENAI_API_KEY"], "openai"),
        ("xai", "grok-4-1-fast", ["XAI_API_KEY", "XAI_KEY"], "httpx"),
        ("google", "gemini-flash-latest", ["GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"], "httpx"),
    ],
)
def test_complete_raises_without_api_key(monkeypatch, provider_name, model, env_vars, import_check):
    if import_check and not _is_installed(import_check):
        pytest.skip(f"Required client library '{import_check}' not installed")
    
    # Ensure env vars are absent
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    
    provider = get_provider(provider_name, model=model, api_key=None)
    
    async def _call():
        await provider.complete([{"role": "user", "content": "ping"}])
    
    with pytest.raises(ValueError):
        asyncio.run(_call())
