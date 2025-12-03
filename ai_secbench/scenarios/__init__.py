"""
Pre-built scenario packs for AI-SecBench.

Scenario packs are curated collections of challenges for specific evaluation purposes.
"""

from typing import List, Dict, Any, Optional
import json
import os

from ai_secbench.core.challenge import ChallengeSet, Challenge
from ai_secbench.challenges import get_challenge_generator


# Built-in scenario pack definitions
SCENARIO_PACKS = {
    "quick_eval": {
        "description": "Quick evaluation with 6 challenges (2 per type)",
        "challenges_per_type": 2,
        "types": ["cipher", "steganographic", "context_poisoning"],
        "difficulty_distribution": {"easy": 0.5, "medium": 0.5, "hard": 0.0},
    },
    "full_eval": {
        "description": "Full evaluation with 15 challenges (5 per type)",
        "challenges_per_type": 5,
        "types": ["cipher", "steganographic", "context_poisoning"],
        "difficulty_distribution": {"easy": 0.2, "medium": 0.5, "hard": 0.3},
    },
    "cipher_focus": {
        "description": "Cipher-focused evaluation with all cipher subtypes",
        "challenges_per_type": 8,
        "types": ["cipher"],
        "difficulty_distribution": {"easy": 0.25, "medium": 0.5, "hard": 0.25},
    },
    "adversarial": {
        "description": "Adversarial robustness focus (prompt-security, authority spoofing)",
        "challenges_per_type": 4,
        "types": ["cipher", "context_poisoning"],
        "subtypes": {
            "cipher": ["prompt_security", "adversarial_mix"],
            "context_poisoning": ["authority_spoofing", "source_manipulation"],
        },
        "difficulty_distribution": {"easy": 0.0, "medium": 0.3, "hard": 0.7},
    },
    "safety": {
        "description": "Safety-focused evaluation",
        "challenges_per_type": 4,
        "types": ["steganographic", "context_poisoning"],
        "subtypes": {
            "steganographic": ["safety_stego"],
            "context_poisoning": ["authority_spoofing"],
        },
        "difficulty_distribution": {"easy": 0.0, "medium": 0.5, "hard": 0.5},
    },
    "norwegian": {
        "description": "Norwegian language evaluation",
        "challenges_per_type": 3,
        "types": ["cipher", "steganographic", "context_poisoning"],
        "language": "norwegian",
        "difficulty_distribution": {"easy": 0.33, "medium": 0.34, "hard": 0.33},
    },
}


def list_scenario_packs() -> List[str]:
    """List available scenario pack names."""
    return list(SCENARIO_PACKS.keys())


def get_scenario_pack_info(name: str) -> Dict[str, Any]:
    """Get information about a scenario pack."""
    if name not in SCENARIO_PACKS:
        raise ValueError(f"Unknown scenario pack: {name}. Available: {list_scenario_packs()}")
    return SCENARIO_PACKS[name].copy()


def load_scenario_pack(
    name: str,
    seed: Optional[int] = None,
    language: Optional[str] = None,
) -> ChallengeSet:
    """
    Load a pre-defined scenario pack.
    
    Args:
        name: Name of the scenario pack
        seed: Random seed for reproducibility
        language: Override default language
    
    Returns:
        ChallengeSet with the generated challenges
    
    Example:
        challenges = load_scenario_pack("quick_eval", seed=42)
        for challenge in challenges:
            print(challenge.challenge_id)
    """
    if name not in SCENARIO_PACKS:
        raise ValueError(f"Unknown scenario pack: {name}. Available: {list_scenario_packs()}")
    
    pack_config = SCENARIO_PACKS[name]
    
    from ai_secbench.challenges.base import GeneratorConfig
    
    # Build generator config
    gen_config = GeneratorConfig(
        language=language or pack_config.get("language", "english"),
        difficulty_distribution=pack_config.get(
            "difficulty_distribution", 
            {"easy": 0.3, "medium": 0.5, "hard": 0.2}
        ),
    )
    
    challenges = []
    
    for challenge_type in pack_config["types"]:
        generator = get_challenge_generator(challenge_type, gen_config)
        
        # Get subtypes if specified
        subtypes = None
        if "subtypes" in pack_config and challenge_type in pack_config["subtypes"]:
            subtypes = pack_config["subtypes"][challenge_type]
        
        type_challenges = generator.generate(
            n=pack_config["challenges_per_type"],
            language=gen_config.language,
            seed=seed,
            subtypes=subtypes,
        )
        
        challenges.extend(type_challenges)
    
    return ChallengeSet(
        challenges=challenges,
        set_id=f"pack_{name}",
        master_seed=seed,
        description=pack_config["description"],
    )


def save_official_set(
    challenge_set: ChallengeSet,
    version: str,
    output_dir: str = "official_sets",
) -> str:
    """
    Save a challenge set as an official versioned set.
    
    Args:
        challenge_set: The challenge set to save
        version: Version string (e.g., "v1.0")
        output_dir: Directory to save to
    
    Returns:
        Path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    challenge_set.set_id = f"official_{version}"
    challenge_set.version = version
    
    output_path = os.path.join(output_dir, f"official_{version}.json")
    challenge_set.save(output_path)
    
    return output_path


def load_official_set(version: str, search_dirs: Optional[List[str]] = None) -> ChallengeSet:
    """
    Load an official versioned challenge set.
    
    Args:
        version: Version string (e.g., "v1.0")
        search_dirs: Directories to search (default: ["official_sets", package data])
    
    Returns:
        ChallengeSet
    """
    if search_dirs is None:
        search_dirs = [
            "official_sets",
            os.path.join(os.path.dirname(__file__), "data"),
        ]
    
    for dir_path in search_dirs:
        file_path = os.path.join(dir_path, f"official_{version}.json")
        if os.path.exists(file_path):
            return ChallengeSet.load(file_path)
    
    raise FileNotFoundError(
        f"Official set version {version} not found in: {search_dirs}"
    )


__all__ = [
    "list_scenario_packs",
    "get_scenario_pack_info",
    "load_scenario_pack",
    "save_official_set",
    "load_official_set",
    "SCENARIO_PACKS",
]
