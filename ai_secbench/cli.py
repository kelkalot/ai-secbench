"""
Command-line entry point for AI-SecBench.
"""

import argparse
import os

from ai_secbench import (
    BenchmarkRunner,
    BenchmarkConfig,
    run_benchmark,
)
from ai_secbench.scenarios import load_scenario_pack, list_scenario_packs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI-SecBench evaluations from the CLI.")
    
    parser.add_argument("--provider", default="anthropic", help="Provider name (anthropic, openai, huggingface, xai, ...)")
    parser.add_argument("--model", default=None, help="Model name/ID for provider")
    parser.add_argument("--judge-provider", default=None, help="Judge provider (defaults to provider)")
    parser.add_argument("--judge-model", default=None, help="Judge model (defaults to model)")
    parser.add_argument("--api-key", default=None, help="API key for main provider (otherwise env var)")
    parser.add_argument("--judge-api-key", default=None, help="API key for judge provider (otherwise env var)")
    
    parser.add_argument("--challenge-types", nargs="+", default=None, help="Challenge types to include")
    parser.add_argument("--n-per-type", type=int, default=2, help="Number of challenges per type")
    parser.add_argument("--language", default="english", help="Language for generated challenges")
    parser.add_argument("--scenario", default=None, help=f"Scenario pack to use (choices: {', '.join(list_scenario_packs())})")
    parser.add_argument("--master-seed", type=int, default=None, help="Seed for reproducibility")
    parser.add_argument("--turn-mode", default="single_shot", choices=["single_shot", "fixed_multi", "interactive"], help="Conversation mode")
    parser.add_argument("--max-turns", type=int, default=5, help="Max turns for multi-turn modes")
    
    parser.add_argument("--require-structured-output", dest="require_structured_output", action="store_true", default=True, help="Require JSON output format in prompts")
    parser.add_argument("--no-require-structured-output", dest="require_structured_output", action="store_false", help="Disable JSON formatting instructions")
    parser.add_argument("--use-official-set", action="store_true", help="Use versioned official set if available")
    parser.add_argument("--official-set-version", default="v1.0", help="Official set version")
    
    parser.add_argument("--output", default=None, help="Path to save results JSON")
    parser.add_argument("--verbose", action="store_true", default=False, help="Verbose logging during run")
    
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    
    if args.scenario and args.challenge_types:
        raise SystemExit("Specify either --scenario or --challenge-types, not both.")
    
    # Scenario pack or generated challenges
    challenge_set = None
    challenge_types = args.challenge_types
    if args.scenario:
        challenge_set = load_scenario_pack(
            args.scenario,
            seed=args.master_seed,
            language=args.language,
        )
        challenge_types = [c.challenge_type for c in challenge_set]
    
    config = BenchmarkConfig(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        judge_provider=args.judge_provider or args.provider,
        judge_model=args.judge_model or args.model or "",
        judge_api_key=args.judge_api_key,
        challenge_types=challenge_types or ["cipher", "steganographic", "context_poisoning"],
        n_challenges_per_type=args.n_per_type,
        language=args.language,
        master_seed=args.master_seed,
        use_official_set=args.use_official_set,
        official_set_version=args.official_set_version,
        turn_mode=args.turn_mode,
        max_turns=args.max_turns,
        require_structured_output=args.require_structured_output,
        verbose=args.verbose,
    )
    
    runner = BenchmarkRunner(config, challenge_set=challenge_set)
    results = runner.run_sync()
    
    print(results.summary())
    
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        results.save(args.output)
        if args.verbose:
            print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
