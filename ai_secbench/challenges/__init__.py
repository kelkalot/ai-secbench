"""
Challenge generators for AI-SecBench.

Supports:
- Cipher challenges (Caesar, Vigenère, adversarial mix, prompt-security)
- Steganographic challenges (acrostics, positional encoding, safety-aware)
- Context poisoning challenges (false facts, contradictions, authority spoofing)

Easily extensible with new challenge types.
"""

from typing import List, Dict, Type, Optional

from ai_secbench.challenges.base import BaseChallengeGenerator, GeneratorConfig
from ai_secbench.challenges.cipher import CipherChallengeGenerator, create_cipher_generator
from ai_secbench.challenges.steganographic import SteganographicChallengeGenerator, create_steganographic_generator
from ai_secbench.challenges.context_poisoning import ContextPoisoningChallengeGenerator, create_context_poisoning_generator


# Registry of challenge generators
_GENERATORS: Dict[str, Type[BaseChallengeGenerator]] = {
    "cipher": CipherChallengeGenerator,
    "steganographic": SteganographicChallengeGenerator,
    "stego": SteganographicChallengeGenerator,  # Alias
    "context_poisoning": ContextPoisoningChallengeGenerator,
    "poisoning": ContextPoisoningChallengeGenerator,  # Alias
}


def list_challenge_types() -> List[str]:
    """List available challenge types."""
    return ["cipher", "steganographic", "context_poisoning"]


def get_challenge_generator(
    challenge_type: str,
    config: Optional[GeneratorConfig] = None,
) -> BaseChallengeGenerator:
    """
    Get a challenge generator by type.
    
    Args:
        challenge_type: Type of challenges ("cipher", "steganographic", "context_poisoning")
        config: Optional generator configuration
    
    Returns:
        Challenge generator instance
    
    Example:
        gen = get_challenge_generator("cipher")
        challenges = gen.generate(n=5, language="english")
    """
    challenge_type = challenge_type.lower()
    
    if challenge_type not in _GENERATORS:
        available = ", ".join(list_challenge_types())
        raise ValueError(
            f"Unknown challenge type: {challenge_type}. Available: {available}"
        )
    
    generator_class = _GENERATORS[challenge_type]
    return generator_class(config or GeneratorConfig())


def register_challenge_type(
    name: str,
    generator_class: Type[BaseChallengeGenerator],
) -> None:
    """
    Register a new challenge type.
    
    Use this to extend AI-SecBench with custom challenge generators.
    
    Args:
        name: Name for the challenge type
        generator_class: Generator class (must inherit from BaseChallengeGenerator)
    
    Example:
        class MyCustomGenerator(BaseChallengeGenerator):
            ...
        
        register_challenge_type("custom", MyCustomGenerator)
    """
    if not issubclass(generator_class, BaseChallengeGenerator):
        raise TypeError(
            f"Generator class must inherit from BaseChallengeGenerator, "
            f"got {generator_class.__name__}"
        )
    
    _GENERATORS[name.lower()] = generator_class


__all__ = [
    # Base
    "BaseChallengeGenerator",
    "GeneratorConfig",
    # Generators
    "CipherChallengeGenerator",
    "SteganographicChallengeGenerator",
    "ContextPoisoningChallengeGenerator",
    # Factory functions
    "create_cipher_generator",
    "create_steganographic_generator",
    "create_context_poisoning_generator",
    # Registry
    "list_challenge_types",
    "get_challenge_generator",
    "register_challenge_type",
]
