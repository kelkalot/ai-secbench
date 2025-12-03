"""
AI-SecBench: Security-Focused AI Reasoning Benchmark

A benchmark for evaluating AI models on security-adjacent reasoning tasks
including cipher analysis, steganographic detection, and adversarial robustness.
"""

__version__ = "0.1.0"
__author__ = "AI-SecBench Team"

from ai_secbench.core.challenge import Challenge, ChallengeResult, ChallengeSet
from ai_secbench.core.runner import BenchmarkRunner, run_benchmark
from ai_secbench.core.evaluator import Evaluator, EvaluationResult
from ai_secbench.core.config import BenchmarkConfig, TurnMode, JudgeMode, CostEstimate

from ai_secbench.providers import get_provider, list_providers
from ai_secbench.challenges import list_challenge_types, get_challenge_generator
from ai_secbench.scenarios import list_scenario_packs, load_scenario_pack

__all__ = [
    # Version
    "__version__",
    # Core
    "Challenge",
    "ChallengeResult", 
    "ChallengeSet",
    "BenchmarkRunner",
    "run_benchmark",
    "Evaluator",
    "EvaluationResult",
    "BenchmarkConfig",
    "TurnMode",
    "JudgeMode",
    "CostEstimate",
    # Providers
    "get_provider",
    "list_providers",
    # Challenges
    "list_challenge_types",
    "get_challenge_generator",
    # Scenarios
    "list_scenario_packs",
    "load_scenario_pack",
]
