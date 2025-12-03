"""
Anthropic (Claude) provider implementation.
"""

import os
from typing import List, Dict, Any, Optional, Tuple, Union

from ai_secbench.providers.base import BaseProvider, ProviderConfig


class AnthropicProvider(BaseProvider):
    """
    Provider for Anthropic's Claude models.
    
    Requires: pip install anthropic
    """
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None
    
    @property
    def client(self):
        """Lazy-load the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package is required for AnthropicProvider. "
                    "Install with: pip install anthropic"
                )
            
            api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "Anthropic API key required. Set ANTHROPIC_API_KEY env var "
                    "or pass api_key to config."
                )
            
            self._client = anthropic.AsyncAnthropic(
                api_key=api_key,
                timeout=self.config.timeout,
            )
        
        return self._client
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        return_usage: bool = False,
        **kwargs,
    ) -> Union[Tuple[str, Dict[str, int]], str]:
        """Send completion request to Claude."""
        
        # Defensive: handle case where a string is passed instead of messages array
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # Extract system message if present
        system_content = None
        chat_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                chat_messages.append(msg)
        
        # Build request
        request_kwargs = {
            "model": self.config.model,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "messages": chat_messages,
        }
        
        if system_content:
            request_kwargs["system"] = system_content
        
        # Make request
        response = await self.client.messages.create(**request_kwargs)
        
        # Extract response text
        response_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                response_text += block.text
        
        if return_usage:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            return response_text, usage
        
        return response_text
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Claude uses a similar tokenizer to GPT - roughly 4 chars per token
        return len(text) // 4


def create_anthropic_provider(
    model: str = "claude-3-5-sonnet-20241022",
    api_key: Optional[str] = None,
    **kwargs,
) -> AnthropicProvider:
    """Factory function to create an Anthropic provider."""
    config = ProviderConfig(
        model=model,
        api_key=api_key,
        **kwargs,
    )
    return AnthropicProvider(config)
