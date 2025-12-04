"""
Benchmark runner - orchestrates challenge generation, execution, and evaluation.
"""

import asyncio
import time
import uuid
from datetime import datetime
import random
from typing import List, Optional, Dict, Any
import json

from ai_secbench.core.config import BenchmarkConfig, TurnMode, CostEstimate
from ai_secbench.core.challenge import Challenge, ChallengeSet, ChallengeResult, ModelResponse
from ai_secbench.core.evaluator import Evaluator, EvaluationResult


class BenchmarkRunner:
    """
    Main orchestrator for running AI-SecBench evaluations.
    
    Usage:
        runner = BenchmarkRunner(config)
        results = await runner.run()
        print(results.summary())
    """
    
    def __init__(
        self,
        config: BenchmarkConfig,
        provider: Optional[Any] = None,  # Will be created from config if None
        judge_provider: Optional[Any] = None,
        challenge_set: Optional[ChallengeSet] = None,  # Use existing set or generate
    ):
        self.config = config
        self._provider = provider
        self._judge_provider = judge_provider
        self._challenge_set = challenge_set
        self._evaluator: Optional[Evaluator] = None
        self._total_cost = CostEstimate()
    
    @property
    def provider(self):
        """Lazy-load the model provider."""
        if self._provider is None:
            from ai_secbench.providers import get_provider
            self._provider = get_provider(
                self.config.provider,
                model=self.config.model,
                api_key=self.config.api_key,
            )
        return self._provider
    
    @property
    def judge_provider(self):
        """Lazy-load the judge provider."""
        if self._judge_provider is None and self.config.judge_provider:
            from ai_secbench.providers import get_provider
            self._judge_provider = get_provider(
                self.config.judge_provider,
                model=self.config.judge_model,
                api_key=self.config.judge_api_key,
            )
        return self._judge_provider
    
    @property
    def evaluator(self) -> Evaluator:
        """Get or create the evaluator."""
        if self._evaluator is None:
            self._evaluator = Evaluator(
                judge_provider=self.judge_provider,
                judge_mode=self.config.judge_mode,
            )
        return self._evaluator
    
    def _get_challenge_set(self) -> ChallengeSet:
        """Get or generate the challenge set."""
        if self._challenge_set is not None:
            return self._challenge_set
        
        if self.config.use_official_set:
            # Load official versioned set
            return self._load_official_set()
        else:
            # Generate fresh challenges
            return self._generate_challenges()
    
    def _load_official_set(self) -> ChallengeSet:
        """Load a versioned official challenge set."""
        import os
        
        # Look for official sets in package data or specified directory
        search_paths = [
            f"official_sets/{self.config.official_set_version}.json",
            os.path.join(
                os.path.dirname(__file__), 
                "..", "scenarios", 
                f"official_{self.config.official_set_version}.json"
            ),
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return ChallengeSet.load(path)
        
        raise FileNotFoundError(
            f"Official set version {self.config.official_set_version} not found. "
            f"Searched: {search_paths}"
        )
    
    def _generate_challenges(self) -> ChallengeSet:
        """Generate a fresh challenge set."""
        from ai_secbench.challenges import get_challenge_generator
        
        # Derive per-type seeds for reproducibility without forcing identical streams
        seeded_rng = random.Random(self.config.master_seed) if self.config.master_seed is not None else None
        challenges = []
        
        for challenge_type in self.config.challenge_types:
            generator = get_challenge_generator(challenge_type)
            type_seed = seeded_rng.randint(0, 1_000_000_000) if seeded_rng else None
            
            new_challenges = generator.generate(
                n=self.config.n_challenges_per_type,
                language=self.config.language,
                seed=type_seed,
                index_offset=0,
            )
            challenges.extend(new_challenges)
        
        return ChallengeSet(
            challenges=challenges,
            set_id=f"generated_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            master_seed=self.config.master_seed,
            description=f"Generated challenge set with types: {self.config.challenge_types}",
        )
    
    async def _run_single_challenge(
        self, 
        challenge: Challenge,
    ) -> ChallengeResult:
        """Run a single challenge through the model."""
        responses: List[ModelResponse] = []
        
        # Build initial prompt
        prompt = challenge.get_prompt(
            include_format_instructions=self.config.require_structured_output
        )
        
        messages = []
        if self.config.system_prompt_template:
            messages.append({
                "role": "system",
                "content": self.config.system_prompt_template,
            })
        
        messages.append({"role": "user", "content": prompt})
        
        max_turns = 1 if self.config.turn_mode == TurnMode.SINGLE_SHOT else self.config.max_turns
        
        for turn in range(max_turns):
            start_time = time.time()
            
            try:
                raw_response, token_info = await self.provider.complete(
                    messages=messages,
                    return_usage=True,
                )
            except Exception as e:
                # Handle API errors gracefully
                raw_response = f"Error: {str(e)}"
                token_info = {"input_tokens": 0, "output_tokens": 0}
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Parse response
            response = ModelResponse.parse(raw_response, turn=turn + 1)
            response.latency_ms = latency_ms
            response.tokens_used = token_info.get("input_tokens", 0) + token_info.get("output_tokens", 0)
            
            responses.append(response)
            
            # Track costs
            if self.config.track_costs:
                self._total_cost.input_tokens += token_info.get("input_tokens", 0)
                self._total_cost.output_tokens += token_info.get("output_tokens", 0)
            
            # Check if we should continue (for multi-turn modes)
            if self.config.turn_mode == TurnMode.SINGLE_SHOT:
                break
            
            if self.config.turn_mode == TurnMode.FIXED_MULTI:
                # Add response to conversation and continue
                messages.append({"role": "assistant", "content": raw_response})
                
                if turn < max_turns - 1:
                    # Prompt for refinement
                    messages.append({
                        "role": "user",
                        "content": "Please review your answer and refine if needed. "
                                   "If you're confident, you can keep your answer the same."
                    })
            
            elif self.config.turn_mode == TurnMode.INTERACTIVE:
                # Check answer and provide feedback
                correctness, feedback = self.evaluator.evaluate_correctness(
                    challenge, response
                )
                
                if correctness >= 80:
                    # Good enough, stop
                    break
                
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({
                    "role": "user",
                    "content": f"Your answer doesn't appear to be correct. {feedback} "
                               "Please try again with a different approach."
                })
        
        # Evaluate the attempt
        result = await self.evaluator.evaluate(challenge, responses)
        
        return result
    
    async def run(
        self,
        progress_callback: Optional[callable] = None,
    ) -> EvaluationResult:
        """
        Run the full benchmark.
        
        Args:
            progress_callback: Optional callback(current, total, challenge_id) for progress updates
        
        Returns:
            EvaluationResult with all scores and details
        """
        run_id = str(uuid.uuid4())[:8]
        started_at = datetime.utcnow().isoformat() + "Z"
        
        if self.config.verbose:
            print(f"Starting benchmark run {run_id}")
            print(f"Provider: {self.config.provider} / {self.config.model}")
        
        # Get challenges
        challenge_set = self._get_challenge_set()
        
        if self.config.verbose:
            stats = challenge_set.get_statistics()
            print(f"Loaded {stats['total']} challenges")
            print(f"Types: {stats['by_type']}")
        
        # Initialize cost tracking
        self._total_cost = CostEstimate(
            provider=self.config.provider,
            model=self.config.model,
        )
        
        # Run each challenge
        results: List[ChallengeResult] = []
        
        for i, challenge in enumerate(challenge_set):
            if self.config.verbose:
                print(f"  [{i+1}/{len(challenge_set)}] {challenge.challenge_id}: {challenge.subtype}")
            
            if progress_callback:
                progress_callback(i, len(challenge_set), challenge.challenge_id)
            
            result = await self._run_single_challenge(challenge)
            results.append(result)
            
            if self.config.verbose:
                status = "✓" if result.passed else "✗"
                print(f"    {status} Score: {result.composite_score:.1f}")
        
        completed_at = datetime.utcnow().isoformat() + "Z"
        
        # Build evaluation result
        eval_result = EvaluationResult(
            results=results,
            config=self.config.to_dict(),
            total_cost=self._total_cost,
            run_id=run_id,
            model=f"{self.config.provider}/{self.config.model}",
            started_at=started_at,
            completed_at=completed_at,
        )
        
        eval_result.compute_aggregates()
        
        if self.config.verbose:
            print("\n" + eval_result.summary())
        
        # Save if output directory specified
        if self.config.output_dir:
            import os
            os.makedirs(self.config.output_dir, exist_ok=True)
            output_path = os.path.join(
                self.config.output_dir,
                f"results_{run_id}.json"
            )
            eval_result.save(output_path)
            if self.config.verbose:
                print(f"Results saved to: {output_path}")
        
        return eval_result
    
    def run_sync(
        self,
        progress_callback: Optional[callable] = None,
    ) -> EvaluationResult:
        """Synchronous wrapper for run()."""
        return asyncio.run(self.run(progress_callback))


def run_benchmark(
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-20241022",
    challenge_types: Optional[List[str]] = None,
    n_per_type: int = 2,
    turn_mode: str = "single_shot",
    language: str = "english",
    verbose: bool = True,
    **kwargs,
) -> EvaluationResult:
    """
    Convenience function to run a benchmark with common defaults.
    
    Args:
        provider: Model provider ("anthropic", "openai", "huggingface")
        model: Model name/ID
        challenge_types: List of challenge types to include
        n_per_type: Number of challenges per type
        turn_mode: "single_shot", "fixed_multi", or "interactive"
        language: "english" or "norwegian"
        verbose: Print progress
        **kwargs: Additional config options
    
    Returns:
        EvaluationResult
    """
    config = BenchmarkConfig(
        provider=provider,
        model=model,
        challenge_types=challenge_types or ["cipher", "steganographic", "context_poisoning"],
        n_challenges_per_type=n_per_type,
        turn_mode=TurnMode(turn_mode),
        language=language,
        verbose=verbose,
        **kwargs,
    )
    
    runner = BenchmarkRunner(config)
    return runner.run_sync()
