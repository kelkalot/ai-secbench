"""
Google AI Studio (Gemini) provider implementation.

Uses the public REST API (v1beta generateContent endpoint).
"""

import os
from typing import List, Dict, Optional, Tuple, Union

import json

from ai_secbench.providers.base import BaseProvider, ProviderConfig


class GoogleAIProvider(BaseProvider):
    """
    Provider for Google AI Studio / Gemini models.
    
    Requires: httpx (install with: pip install httpx)
    """
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None
        self._base_url = config.base_url or "https://generativelanguage.googleapis.com"
    
    @property
    def client(self):
        """Lazy-load the HTTP client."""
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:
                raise ImportError(
                    "httpx package is required for GoogleAIProvider. Install with: pip install httpx"
                ) from exc
            
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self.config.timeout,
            )
        return self._client
    
    def _messages_to_contents(self, messages: List[Dict[str, str]]) -> List[Dict[str, object]]:
        """Convert OpenAI-style messages to Gemini contents format."""
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            # Gemini uses "model" for assistant responses
            if role == "assistant":
                role = "model"
            parts = [{"text": msg.get("content", "")}]
            contents.append({"role": role, "parts": parts})
        return contents
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        return_usage: bool = False,
        **kwargs,
    ) -> Union[Tuple[str, Dict[str, int]], str]:
        """
        Send completion request to Gemini.
        """
        api_key = (
            self.config.api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GOOGLE_GENAI_API_KEY")
        )
        if not api_key:
            raise ValueError("Google AI API key required. Set GOOGLE_API_KEY env var or pass api_key to config.")
        
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        contents = self._messages_to_contents(messages)
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "maxOutputTokens": kwargs.get("max_tokens", self.config.max_tokens),
            },
        }
        
        path = f"/v1beta/models/{self.config.model}:generateContent?key={api_key}"
        response = await self.client.post(
            path,
            headers={"Content-Type": "application/json"},
            content=json.dumps(payload),
        )
        response.raise_for_status()
        data = response.json()
        
        candidates = data.get("candidates") or []
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            # concatenate text parts
            text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        
        if return_usage:
            usage = data.get("usageMetadata", {}) or {}
            usage_out = {
                "input_tokens": usage.get("promptTokenCount", 0),
                "output_tokens": usage.get("candidatesTokenCount", 0),
            }
            return text, usage_out
        
        return text
    
    def count_tokens(self, text: str) -> int:
        """Rough token estimate."""
        return len(text) // 4


def create_google_provider(
    model: str = "gemini-flash-latest",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> GoogleAIProvider:
    """Factory to create a Google AI (Gemini) provider."""
    config = ProviderConfig(
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )
    return GoogleAIProvider(config)
