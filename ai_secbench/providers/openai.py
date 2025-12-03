"""
OpenAI provider implementation.
"""

import os
from typing import List, Dict, Any, Optional, Tuple, Union

from ai_secbench.providers.base import BaseProvider, ProviderConfig


class OpenAIProvider(BaseProvider):
    """
    Provider for OpenAI models (GPT-4, GPT-3.5, etc.).
    
    Requires: pip install openai
    """
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None
    
    @property
    def client(self):
        """Lazy-load the OpenAI client."""
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAIProvider. "
                    "Install with: pip install openai"
                )
            
            api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY env var "
                    "or pass api_key to config."
                )
            
            client_kwargs = {
                "api_key": api_key,
                "timeout": self.config.timeout,
            }
            
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            
            self._client = openai.AsyncOpenAI(**client_kwargs)
        
        return self._client
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        return_usage: bool = False,
        **kwargs,
    ) -> Union[Tuple[str, Dict[str, int]], str]:
        """Send completion request to OpenAI."""
        
        # Defensive: handle case where a string is passed instead of messages array
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        # Build request
        request_kwargs = {
            "model": self.config.model,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "messages": messages,
        }
        
        # Make request
        response = await self.client.chat.completions.create(**request_kwargs)
        
        # Extract response text
        response_text = response.choices[0].message.content or ""
        
        if return_usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
            return response_text, usage
        
        return response_text
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count."""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.config.model)
            return len(encoding.encode(text))
        except ImportError:
            # Fallback: rough approximation
            return len(text) // 4


def create_openai_provider(
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> OpenAIProvider:
    """Factory function to create an OpenAI provider."""
    config = ProviderConfig(
        model=model,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )
    return OpenAIProvider(config)
