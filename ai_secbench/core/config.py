"""
Benchmark configuration and settings.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class TurnMode(Enum):
    """How many turns the model gets to solve a challenge."""
    SINGLE_SHOT = "single_shot"      # One turn only
    FIXED_MULTI = "fixed_multi"      # Fixed N turns
    INTERACTIVE = "interactive"       # Adaptive with feedback


class JudgeMode(Enum):
    """How reasoning is evaluated."""
    AUTOMATIC = "automatic"           # Rule-based + LLM judge
    RUBRIC_ONLY = "rubric_only"       # LLM judge with rubric
    CORRECTNESS_ONLY = "correctness"  # Only check final answer


@dataclass
class CostEstimate:
    """Tracks estimated API costs."""
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = ""
    model: str = ""
    
    @property
    def estimated_cost_usd(self) -> float:
        """Rough cost estimate based on typical pricing."""
        # Approximate costs per 1M tokens (as of 2024)
        pricing = {
            "anthropic": {"claude-3-5-sonnet": (3.0, 15.0), "claude-3-opus": (15.0, 75.0), "claude-3-haiku": (0.25, 1.25)},
            "openai": {"gpt-4o": (2.5, 10.0), "gpt-4-turbo": (10.0, 30.0), "gpt-3.5-turbo": (0.5, 1.5)},
            "huggingface": {"default": (0.0, 0.0)},  # Often free tier
        }
        
        provider_pricing = pricing.get(self.provider.lower(), {})
        
        # Find matching model or use default
        model_key = None
        for key in provider_pricing:
            if key in self.model.lower():
                model_key = key
                break
        
        if model_key is None:
            model_key = list(provider_pricing.keys())[0] if provider_pricing else None
        
        if model_key is None:
            return 0.0
            
        input_rate, output_rate = provider_pricing[model_key]
        cost = (self.input_tokens / 1_000_000 * input_rate) + (self.output_tokens / 1_000_000 * output_rate)
        return round(cost, 4)
    
    def __add__(self, other: "CostEstimate") -> "CostEstimate":
        return CostEstimate(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            provider=self.provider or other.provider,
            model=self.model or other.model,
        )


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    
    # Provider settings
    provider: str = "anthropic"
    model: str = "claude-3-5-sonnet-20241022"
    api_key: Optional[str] = None  # Falls back to env var
    
    # Judge settings
    judge_provider: str = "anthropic"
    judge_model: str = "claude-3-5-sonnet-20241022"
    judge_api_key: Optional[str] = None
    judge_mode: JudgeMode = JudgeMode.AUTOMATIC
    
    # Turn settings
    turn_mode: TurnMode = TurnMode.SINGLE_SHOT
    max_turns: int = 5
    
    # Challenge settings
    challenge_types: List[str] = field(default_factory=lambda: ["cipher", "steganographic", "context_poisoning"])
    n_challenges_per_type: int = 2
    language: str = "english"  # or "norwegian", affects plaintext pool
    
    # Generation settings
    master_seed: Optional[int] = None  # None = random each run
    use_official_set: bool = False     # Use pre-generated official challenges
    official_set_version: str = "v1.0"
    
    # Output settings
    verbose: bool = True
    save_reasoning: bool = True
    track_costs: bool = True
    output_dir: Optional[str] = None
    
    # Prompt settings
    system_prompt_template: Optional[str] = None
    require_structured_output: bool = True  # Force JSON response format
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "provider": self.provider,
            "model": self.model,
            "judge_provider": self.judge_provider,
            "judge_model": self.judge_model,
            "judge_mode": self.judge_mode.value,
            "turn_mode": self.turn_mode.value,
            "max_turns": self.max_turns,
            "challenge_types": self.challenge_types,
            "n_challenges_per_type": self.n_challenges_per_type,
            "language": self.language,
            "master_seed": self.master_seed,
            "use_official_set": self.use_official_set,
            "official_set_version": self.official_set_version,
            "verbose": self.verbose,
            "save_reasoning": self.save_reasoning,
            "track_costs": self.track_costs,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkConfig":
        """Create from dictionary."""
        if "judge_mode" in d and isinstance(d["judge_mode"], str):
            d["judge_mode"] = JudgeMode(d["judge_mode"])
        if "turn_mode" in d and isinstance(d["turn_mode"], str):
            d["turn_mode"] = TurnMode(d["turn_mode"])
        return cls(**d)
