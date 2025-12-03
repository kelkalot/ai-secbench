"""
XAI (Grok) provider implementation.

Targets xAI's OpenAI-compatible Chat Completions API for Grok models.
"""

import os
from typing import List, Dict, Any, Optional, Tuple, Union

import json

from ai_secbench.providers.base import BaseProvider, ProviderConfig


class XAIProvider(BaseProvider):
    """
    Provider for xAI Grok models via the OpenAI-compatible HTTP API.
    
    Requires: httpx (install with `pip install httpx`)
    """
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None
        self._base_url = config.base_url or "https://api.x.ai/v1"
    
    @property
    def client(self):
        """Lazy-load the HTTP client."""
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:
                raise ImportError(
                    "httpx package is required for XAIProvider. "
                    "Install with: pip install httpx"
                ) from exc
            
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self.config.timeout,
            )
        
        return self._client
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        return_usage: bool = False,
        **kwargs,
    ) -> Union[Tuple[str, Dict[str, int]], str]:
        """
        Send completion request to xAI Grok (Chat Completions format).
        """
        api_key = self.config.api_key or os.environ.get("XAI_API_KEY") or os.environ.get("XAI_KEY")
        if not api_key:
            raise ValueError("XAI API key required. Set XAI_API_KEY env var or pass api_key to config.")
        
        # Defensive: handle string input
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        response = await self.client.post(
            "/chat/completions",
            headers=headers,
            content=json.dumps(payload),
        )
        response.raise_for_status()
        data = response.json()
        
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        
        if return_usage:
            usage = data.get("usage", {}) or {}
            usage_out = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }
            return content, usage_out
        
        return content
    
    def count_tokens(self, text: str) -> int:
        """Rough token estimate (Grok uses GPT-style tokenization)."""
        return len(text) // 4


def create_xai_provider(
    model: str = "grok-2-1212",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> XAIProvider:
    """Factory to create an xAI Grok provider."""
    config = ProviderConfig(
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )
    return XAIProvider(config)
