"""
Tests for AI-SecBench challenge generation.
"""

import pytest
from ai_secbench.challenges import (
    get_challenge_generator,
    list_challenge_types,
    CipherChallengeGenerator,
    SteganographicChallengeGenerator,
    ContextPoisoningChallengeGenerator,
)
from ai_secbench.challenges.base import GeneratorConfig
from ai_secbench.core.challenge import Challenge, ChallengeSet


class TestChallengeRegistry:
    """Test challenge type registry."""
    
    def test_list_types(self):
        types = list_challenge_types()
        assert "cipher" in types
        assert "steganographic" in types
        assert "context_poisoning" in types
    
    def test_get_cipher_generator(self):
        gen = get_challenge_generator("cipher")
        assert isinstance(gen, CipherChallengeGenerator)
    
    def test_get_steganographic_generator(self):
        gen = get_challenge_generator("steganographic")
        assert isinstance(gen, SteganographicChallengeGenerator)
    
    def test_get_context_poisoning_generator(self):
        gen = get_challenge_generator("context_poisoning")
        assert isinstance(gen, ContextPoisoningChallengeGenerator)
    
    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            get_challenge_generator("invalid_type")


class TestCipherGenerator:
    """Test cipher challenge generation."""
    
    def test_generate_single(self):
        gen = get_challenge_generator("cipher")
        challenges = gen.generate(n=1, seed=42)
        
        assert len(challenges) == 1
        assert isinstance(challenges[0], Challenge)
        assert challenges[0].challenge_type == "cipher"
    
    def test_generate_multiple(self):
        gen = get_challenge_generator("cipher")
        challenges = gen.generate(n=4, seed=42)
        
        assert len(challenges) == 4
        # Should cycle through subtypes
        subtypes = [c.subtype for c in challenges]
        assert len(set(subtypes)) >= 2
    
    def test_reproducibility(self):
        gen = get_challenge_generator("cipher")
        
        c1 = gen.generate(n=2, seed=42)
        c2 = gen.generate(n=2, seed=42)
        
        assert c1[0].task_card == c2[0].task_card
        assert c1[1].task_card == c2[1].task_card
    
    def test_language_english(self):
        gen = get_challenge_generator("cipher")
        challenges = gen.generate(n=1, language="english", seed=42)
        
        assert challenges[0].language == "english"
    
    def test_language_norwegian(self):
        gen = get_challenge_generator("cipher")
        challenges = gen.generate(n=1, language="norwegian", seed=42)
        
        assert challenges[0].language == "norwegian"
    
    def test_subtypes(self):
        gen = get_challenge_generator("cipher")
        
        expected_subtypes = ["multi_shift_caesar", "vigenere_autokey", "adversarial_mix", "prompt_security"]
        assert set(gen.subtypes) == set(expected_subtypes)
    
    def test_challenge_has_required_fields(self):
        gen = get_challenge_generator("cipher")
        challenges = gen.generate(n=1, seed=42)
        c = challenges[0]
        
        assert c.challenge_id
        assert c.challenge_type
        assert c.subtype
        assert c.difficulty in ["easy", "medium", "hard"]
        assert c.task_card
        assert c.solution_bundle
        assert "plaintext" in c.solution_bundle or "answer" in c.solution_bundle


class TestSteganographicGenerator:
    """Test steganographic challenge generation."""
    
    def test_generate_single(self):
        gen = get_challenge_generator("steganographic")
        challenges = gen.generate(n=1, seed=42)
        
        assert len(challenges) == 1
        assert challenges[0].challenge_type == "steganographic"
    
    def test_acrostic_pattern(self):
        gen = get_challenge_generator("steganographic")
        challenges = gen.generate(n=1, subtypes=["acrostic_lines"], seed=42)
        
        c = challenges[0]
        assert c.subtype == "acrostic_lines"
        assert "hidden_message" in c.solution_bundle
    
    def test_safety_stego(self):
        gen = get_challenge_generator("steganographic")
        challenges = gen.generate(n=1, subtypes=["safety_stego"], seed=42)
        
        c = challenges[0]
        assert c.subtype == "safety_stego"
        assert "is_harmful" in c.solution_bundle


class TestContextPoisoningGenerator:
    """Test context poisoning challenge generation."""
    
    def test_generate_single(self):
        gen = get_challenge_generator("context_poisoning")
        challenges = gen.generate(n=1, seed=42)
        
        assert len(challenges) == 1
        assert challenges[0].challenge_type == "context_poisoning"
    
    def test_false_fact(self):
        gen = get_challenge_generator("context_poisoning")
        challenges = gen.generate(n=1, subtypes=["false_fact"], seed=42)
        
        c = challenges[0]
        assert c.subtype == "false_fact"
        assert "correct_answer" in c.solution_bundle
        assert "false_claim_in_context" in c.solution_bundle
    
    def test_authority_spoofing(self):
        gen = get_challenge_generator("context_poisoning")
        challenges = gen.generate(n=1, subtypes=["authority_spoofing"], seed=42)
        
        c = challenges[0]
        assert c.subtype == "authority_spoofing"


class TestChallengeSet:
    """Test ChallengeSet functionality."""
    
    def test_create_set(self):
        gen = get_challenge_generator("cipher")
        challenges = gen.generate(n=3, seed=42)
        
        challenge_set = ChallengeSet(
            challenges=challenges,
            set_id="test_set",
            master_seed=42,
        )
        
        assert len(challenge_set) == 3
        assert challenge_set.set_id == "test_set"
    
    def test_filter_by_type(self):
        gen1 = get_challenge_generator("cipher")
        gen2 = get_challenge_generator("steganographic")
        
        challenges = gen1.generate(n=2, seed=42) + gen2.generate(n=2, seed=42)
        challenge_set = ChallengeSet(challenges=challenges)
        
        cipher_only = challenge_set.filter_by_type("cipher")
        assert len(cipher_only) == 2
        assert all(c.challenge_type == "cipher" for c in cipher_only)
    
    def test_statistics(self):
        gen = get_challenge_generator("cipher")
        challenges = gen.generate(n=4, seed=42)
        challenge_set = ChallengeSet(challenges=challenges)
        
        stats = challenge_set.get_statistics()
        assert stats["total"] == 4
        assert "cipher" in stats["by_type"]
    
    def test_serialization(self, tmp_path):
        gen = get_challenge_generator("cipher")
        challenges = gen.generate(n=2, seed=42)
        challenge_set = ChallengeSet(
            challenges=challenges,
            set_id="test",
            master_seed=42,
        )
        
        # Save
        path = tmp_path / "test_set.json"
        challenge_set.save(str(path))
        
        # Load
        loaded = ChallengeSet.load(str(path))
        
        assert len(loaded) == 2
        assert loaded.set_id == "test"
        assert loaded.master_seed == 42


class TestChallenge:
    """Test Challenge class."""
    
    def test_get_prompt(self):
        gen = get_challenge_generator("cipher")
        challenges = gen.generate(n=1, seed=42)
        c = challenges[0]
        
        prompt = c.get_prompt()
        
        assert "Instructions" in prompt or "instructions" in prompt.lower()
        assert "Response Format" in prompt
    
    def test_compute_hash(self):
        gen = get_challenge_generator("cipher")
        challenges = gen.generate(n=2, seed=42)
        
        h1 = challenges[0].compute_hash()
        h2 = challenges[1].compute_hash()
        
        # Different challenges should have different hashes
        assert h1 != h2
        
        # Same challenge should have same hash
        assert h1 == challenges[0].compute_hash()
