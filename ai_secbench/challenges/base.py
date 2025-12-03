"""
Base challenge generator interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import random
import datetime


@dataclass
class GeneratorConfig:
    """Configuration for challenge generators."""
    language: str = "english"
    difficulty_distribution: Dict[str, float] = None  # {"easy": 0.3, "medium": 0.5, "hard": 0.2}
    include_decoys: bool = True
    include_noise: bool = True
    noise_probability: float = 0.08
    
    def __post_init__(self):
        if self.difficulty_distribution is None:
            self.difficulty_distribution = {"easy": 0.3, "medium": 0.5, "hard": 0.2}


class BaseChallengeGenerator(ABC):
    """
    Abstract base class for challenge generators.
    
    Each generator must implement:
    - generate(): Create new challenges
    - challenge_type: Return the type identifier
    - subtypes: Return list of supported subtypes
    """
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
    
    @property
    @abstractmethod
    def challenge_type(self) -> str:
        """Return the challenge type identifier (e.g., 'cipher')."""
        pass
    
    @property
    @abstractmethod
    def subtypes(self) -> List[str]:
        """Return list of supported subtypes."""
        pass
    
    @abstractmethod
    def generate(
        self,
        n: int = 1,
        language: Optional[str] = None,
        seed: Optional[int] = None,
        index_offset: int = 0,
        subtypes: Optional[List[str]] = None,
    ) -> List["Challenge"]:
        """
        Generate n challenges.
        
        Args:
            n: Number of challenges to generate
            language: Override config language
            seed: Random seed for reproducibility
            index_offset: Offset for challenge IDs
            subtypes: Specific subtypes to generate (None = all)
        
        Returns:
            List of Challenge objects
        """
        pass
    
    def _select_difficulty(self) -> str:
        """Select difficulty based on distribution."""
        r = random.random()
        cumulative = 0.0
        for difficulty, prob in self.config.difficulty_distribution.items():
            cumulative += prob
            if r < cumulative:
                return difficulty
        return "medium"
    
    def _make_challenge_id(self, subtype: str, index: int) -> str:
        """Generate a unique challenge ID."""
        type_abbrev = self.challenge_type[:3].upper()
        subtype_abbrev = subtype[:4].upper()
        return f"{type_abbrev}-{subtype_abbrev}-{index:03d}"


# Import Challenge here to avoid circular import
from ai_secbench.core.challenge import Challenge
