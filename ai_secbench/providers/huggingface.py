"""
HuggingFace provider implementation.

Supports both:
1. HuggingFace Inference API (hosted models)
2. Local transformers models (future extension)
"""

import os
from typing import List, Dict, Any, Optional, Tuple, Union

from ai_secbench.providers.base import BaseProvider, ProviderConfig


class HuggingFaceProvider(BaseProvider):
    """
    Provider for HuggingFace models via Inference API.
    
    Requires: pip install huggingface_hub
    
    For local models, use LocalHuggingFaceProvider instead.
    """
    
    # Known chat models and their formats
    CHAT_MODELS = {
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct", 
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "microsoft/Phi-3-mini-4k-instruct",
        "google/gemma-2-27b-it",
        "Qwen/Qwen2.5-72B-Instruct",
    }
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None
    
    @property
    def client(self):
        """Lazy-load the HuggingFace client."""
        if self._client is None:
            try:
                from huggingface_hub import AsyncInferenceClient
            except ImportError:
                raise ImportError(
                    "huggingface_hub package is required for HuggingFaceProvider. "
                    "Install with: pip install huggingface_hub"
                )
            
            api_key = self.config.api_key or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            
            self._client = AsyncInferenceClient(
                model=self.config.model,
                token=api_key,
                timeout=self.config.timeout,
            )
        
        return self._client
    
    def _format_messages_for_model(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages for models that don't support chat format natively.
        Uses a generic template.
        """
        formatted = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        
        formatted.append("Assistant:")
        return "\n\n".join(formatted)
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        return_usage: bool = False,
        **kwargs,
    ) -> Union[Tuple[str, Dict[str, int]], str]:
        """Send completion request to HuggingFace Inference API."""
        
        # Check if model supports chat format
        is_chat_model = any(
            chat_model.lower() in self.config.model.lower() 
            for chat_model in self.CHAT_MODELS
        ) or "instruct" in self.config.model.lower() or "chat" in self.config.model.lower()
        
        if is_chat_model:
            # Use chat completion
            response = await self.client.chat_completion(
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
            )
            
            response_text = response.choices[0].message.content or ""
            
            if return_usage:
                usage = {
                    "input_tokens": getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
                    "output_tokens": getattr(response.usage, "completion_tokens", 0) if response.usage else 0,
                }
                return response_text, usage
            
            return response_text
        else:
            # Use text generation with formatted prompt
            prompt = self._format_messages_for_model(messages)
            
            response = await self.client.text_generation(
                prompt=prompt,
                max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature) or 0.01,  # HF doesn't like 0
                return_full_text=False,
            )
            
            response_text = response if isinstance(response, str) else response.generated_text
            
            if return_usage:
                # Estimate tokens
                usage = {
                    "input_tokens": len(prompt) // 4,
                    "output_tokens": len(response_text) // 4,
                }
                return response_text, usage
            
            return response_text
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Most HF models use similar tokenization
        return len(text) // 4


class LocalHuggingFaceProvider(BaseProvider):
    """
    Provider for local HuggingFace transformers models.
    
    Requires: pip install transformers torch
    
    Note: This loads the model into memory, which can be slow and memory-intensive.
    """
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Load model and tokenizer."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required for LocalHuggingFaceProvider. "
                "Install with: pip install transformers torch"
            )
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        return_usage: bool = False,
        **kwargs,
    ) -> Union[Tuple[str, Dict[str, int]], str]:
        """Generate completion using local model."""
        import asyncio
        
        # Run in thread pool since model inference is blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._complete_sync,
            messages,
            return_usage,
            kwargs,
        )
        return result
    
    def _complete_sync(
        self,
        messages: List[Dict[str, str]],
        return_usage: bool,
        kwargs: dict,
    ):
        """Synchronous completion."""
        import torch
        
        self._load_model()
        
        # Format messages
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback formatting
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            prompt += "\nassistant:"
        
        # Tokenize
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        input_length = inputs.input_ids.shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature) or 0.01,
                do_sample=kwargs.get("temperature", self.config.temperature) > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        
        # Decode
        response_text = self._tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        )
        
        if return_usage:
            usage = {
                "input_tokens": input_length,
                "output_tokens": outputs.shape[1] - input_length,
            }
            return response_text, usage
        
        return response_text
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer."""
        self._load_model()
        return len(self._tokenizer.encode(text))


def create_huggingface_provider(
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    api_key: Optional[str] = None,
    local: bool = False,
    **kwargs,
) -> Union[HuggingFaceProvider, LocalHuggingFaceProvider]:
    """
    Factory function to create a HuggingFace provider.
    
    Args:
        model: Model ID from HuggingFace Hub
        api_key: HuggingFace token (or set HF_TOKEN env var)
        local: If True, use local model instead of Inference API
        **kwargs: Additional config options
    """
    config = ProviderConfig(
        model=model,
        api_key=api_key,
        **kwargs,
    )
    
    if local:
        return LocalHuggingFaceProvider(config)
    return HuggingFaceProvider(config)
