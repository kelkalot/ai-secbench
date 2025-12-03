#!/usr/bin/env python3
"""
AI-SecBench Example Usage

This script demonstrates various ways to use AI-SecBench:
1. Generate challenges and explore them
2. Run a benchmark evaluation
3. Use scenario packs
4. Save and load challenge sets
"""

import asyncio
import json
from pathlib import Path

# Assuming ai_secbench is installed or in PYTHONPATH
from ai_secbench import (
    BenchmarkConfig,
    BenchmarkRunner,
    TurnMode,
    JudgeMode,
    ChallengeSet,
)
from ai_secbench.challenges import (
    get_challenge_generator,
    list_challenge_types,
)
from ai_secbench.scenarios import (
    load_scenario_pack,
    list_scenario_packs,
)
from ai_secbench.utils import (
    caesar_encrypt,
    vigenere_encrypt,
    random_plaintext,
)


def demo_challenge_generation():
    """Demonstrate challenge generation."""
    print("=" * 60)
    print("DEMO: Challenge Generation")
    print("=" * 60)
    
    # List available types
    print(f"\nAvailable challenge types: {list_challenge_types()}")
    
    # Generate cipher challenges
    print("\n--- Cipher Challenges ---")
    cipher_gen = get_challenge_generator("cipher")
    print(f"Cipher subtypes: {cipher_gen.subtypes}")
    
    cipher_challenges = cipher_gen.generate(n=2, language="english", seed=42)
    
    for c in cipher_challenges:
        print(f"\n{c.challenge_id} ({c.subtype}, {c.difficulty})")
        print(f"  Solution: {c.solution_bundle.get('plaintext', 'N/A')[:50]}...")
    
    # Generate steganographic challenges
    print("\n--- Steganographic Challenges ---")
    stego_gen = get_challenge_generator("steganographic")
    print(f"Stego subtypes: {stego_gen.subtypes}")
    
    stego_challenges = stego_gen.generate(n=2, seed=42)
    
    for c in stego_challenges:
        print(f"\n{c.challenge_id} ({c.subtype}, {c.difficulty})")
        print(f"  Hidden message: {c.solution_bundle.get('hidden_message', 'N/A')}")
    
    # Generate context poisoning challenges
    print("\n--- Context Poisoning Challenges ---")
    poison_gen = get_challenge_generator("context_poisoning")
    print(f"Poisoning subtypes: {poison_gen.subtypes}")
    
    poison_challenges = poison_gen.generate(n=2, seed=42)
    
    for c in poison_challenges:
        print(f"\n{c.challenge_id} ({c.subtype}, {c.difficulty})")
        print(f"  Correct answer: {c.solution_bundle.get('correct_answer', 'N/A')}")


def demo_view_challenge():
    """Show what a challenge looks like to the model."""
    print("\n" + "=" * 60)
    print("DEMO: Challenge Prompt (What the Model Sees)")
    print("=" * 60)
    
    gen = get_challenge_generator("cipher")
    challenges = gen.generate(n=1, subtypes=["multi_shift_caesar"], seed=42)
    c = challenges[0]
    
    print(f"\nChallenge: {c.challenge_id}")
    print("-" * 40)
    print(c.get_prompt())
    print("-" * 40)
    print(f"\n[HIDDEN FROM MODEL] Solution:")
    print(json.dumps(c.solution_bundle, indent=2))


def demo_scenario_packs():
    """Demonstrate scenario pack usage."""
    print("\n" + "=" * 60)
    print("DEMO: Scenario Packs")
    print("=" * 60)
    
    print(f"\nAvailable packs: {list_scenario_packs()}")
    
    # Load quick_eval pack
    print("\nLoading 'quick_eval' pack...")
    challenges = load_scenario_pack("quick_eval", seed=42)
    
    stats = challenges.get_statistics()
    print(f"  Total challenges: {stats['total']}")
    print(f"  By type: {stats['by_type']}")
    print(f"  By difficulty: {stats['by_difficulty']}")


def demo_cipher_utilities():
    """Demonstrate cipher utility functions."""
    print("\n" + "=" * 60)
    print("DEMO: Cipher Utilities")
    print("=" * 60)
    
    plaintext = random_plaintext("english")
    print(f"\nPlaintext: {plaintext}")
    
    # Caesar
    caesar = caesar_encrypt(plaintext, 3)
    print(f"Caesar (k=3): {caesar}")
    
    # Vigenère
    key = "secret"
    vigenere = vigenere_encrypt(plaintext, key)
    print(f"Vigenère (key={key}): {vigenere}")


def demo_save_load():
    """Demonstrate saving and loading challenge sets."""
    print("\n" + "=" * 60)
    print("DEMO: Save/Load Challenge Sets")
    print("=" * 60)
    
    # Generate a challenge set
    gen = get_challenge_generator("cipher")
    challenges = gen.generate(n=3, seed=42)
    
    challenge_set = ChallengeSet(
        challenges=challenges,
        set_id="demo_set",
        version="1.0",
        master_seed=42,
        description="Demo challenge set",
    )
    
    # Save to file
    output_path = Path("demo_challenges.json")
    challenge_set.save(str(output_path))
    print(f"\nSaved to: {output_path}")
    
    # Load from file
    loaded = ChallengeSet.load(str(output_path))
    print(f"Loaded: {loaded.set_id} with {len(loaded)} challenges")
    
    # Clean up
    output_path.unlink()
    print("Cleaned up demo file")


async def demo_benchmark_run():
    """
    Demonstrate running a benchmark.
    
    NOTE: This requires API keys to be set!
    Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.
    """
    print("\n" + "=" * 60)
    print("DEMO: Benchmark Run (requires API key)")
    print("=" * 60)
    
    # Check for API key
    import os
    has_anthropic = os.environ.get("ANTHROPIC_API_KEY")
    has_openai = os.environ.get("OPENAI_API_KEY")
    
    if not has_anthropic and not has_openai:
        print("\nSkipping benchmark demo - no API key found.")
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY to run this demo.")
        return
    
    provider = "anthropic" if has_anthropic else "openai"
    model = "claude-3-5-sonnet-20241022" if has_anthropic else "gpt-4o"
    
    print(f"\nUsing provider: {provider}")
    print(f"Model: {model}")
    
    config = BenchmarkConfig(
        provider=provider,
        model=model,
        challenge_types=["cipher"],  # Just cipher for demo
        n_challenges_per_type=1,
        turn_mode=TurnMode.SINGLE_SHOT,
        judge_mode=JudgeMode.CORRECTNESS_ONLY,  # Skip LLM judge for demo
        master_seed=42,
        verbose=True,
        track_costs=True,
    )
    
    runner = BenchmarkRunner(config)
    
    print("\nRunning benchmark...")
    results = await runner.run()
    
    print("\n" + results.summary())


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("AI-SecBench Demo Suite")
    print("=" * 60)
    
    # Run synchronous demos
    demo_challenge_generation()
    demo_view_challenge()
    demo_scenario_packs()
    demo_cipher_utilities()
    demo_save_load()
    
    # Run async benchmark demo
    asyncio.run(demo_benchmark_run())
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
