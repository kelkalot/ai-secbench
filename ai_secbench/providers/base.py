"""
Base provider interface for model integrations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Common configuration for providers."""
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 120.0
    max_tokens: int = 4096
    temperature: float = 0.0  # Deterministic for benchmarking


class BaseProvider(ABC):
    """
    Abstract base class for model providers.
    
    All providers must implement the complete() method.
    """
    
    def __init__(self, config: ProviderConfig):
        self.config = config
    
    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        return_usage: bool = False,
        **kwargs,
    ) -> Union[Tuple[str, Dict[str, int]], str]:
        """
        Send a completion request to the model.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            return_usage: If True, return (response, usage_dict)
            **kwargs: Provider-specific options
        
        Returns:
            If return_usage: (response_text, {"input_tokens": N, "output_tokens": M})
            Otherwise: response_text
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Estimate token count for a text string."""
        pass
    
    @property
    def name(self) -> str:
        """Provider name for logging."""
        return self.__class__.__name__.replace("Provider", "").lower()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model})"
